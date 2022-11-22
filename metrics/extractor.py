import torch
import torch.nn as nn
import torchvision.transforms as transforms
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .inception import InceptionV3
import numpy as np
from PIL import Image


model_versions = {"InceptionV3_torch": "pytorch/vision:v0.10.0",
                  "ResNet_torch": "pytorch/vision:v0.10.0",
                  "SwAV_torch": "facebookresearch/swav:main"}
model_names = {"InceptionV3_torch": "inception_v3",
               "ResNet50_torch": "resnet50",
               "SwAV_torch": "resnet50"}

SWAV_CLASSIFIER_URL = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar"

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []
        

class Extractor(nn.Module):
    def __init__(self, backbone, post_resizer, device):
        super().__init__()
        self.eval_backbone = backbone
        self.post_resizer = post_resizer
        self.device = device
        self.save_output = SaveOutput()
        
        if self.eval_backbone == "InceptionV3_tf":
            self.res, mean, std = 299, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            self.model = InceptionV3(output_blocks=[3], resize_input=False, normalize_input=False).to(device)
        # TODO
        elif self.eval_backbone in ["InceptionV3_torch", "ResNet50_torch", "SwAV_torch"]:
            self.res = 299 if "InceptionV3" in self.eval_backbone else 224
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.model = torch.hub.load(model_versions[self.eval_backbone],
                                        model_names[self.eval_backbone],
                                        pretrained=True)
            if self.eval_backbone == "SwAV_torch":
               linear_state_dict = load_state_dict_from_url(SWAV_CLASSIFIER_URL, progress=True)["state_dict"]
               linear_state_dict = {k.replace("module.linear.", ""): v for k, v in linear_state_dict.items()}
               self.model.fc.load_state_dict(linear_state_dict, strict=True)

            self.model = self.model.to(self.device)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)
                    
        else:
            raise NotImplementedError


        self.resizer = build_resizer(resizer=self.post_resizer, backbone=self.eval_backbone, size=self.res)
        self.totensor = transforms.ToTensor()
        # .to(self.device) : accelerator의 device로 올라가는지 확인
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)

        # requrire grade=True로 설정?
        # misc.make_model_require_grad(self.model)

    def eval(self):
        self.model.eval()

    def get_outputs(self, x, quantize=False):
        if quantize:
            x = quantize_images(x)
        else:
            x = x.detach().cpu().numpy().astype(np.uint8)
        x = resize_images(x, self.resizer, self.totensor, self.mean, self.std, device=self.device)
        repres, logits = self.model(x)
        return repres, logits


dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    }
}

def build_resizer(resizer, backbone, size):
    if resizer == "friendly":
        if backbone == "InceptionV3_tf":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "InceptionV3_torch":
            return make_resizer("PIL", "lanczos", (size, size))
        elif backbone == "ResNet50_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "SwAV_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "DINO_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "Swin-T_torch":
            return make_resizer("PIL", "bicubic", (size, size))
        else:
            raise ValueError(f"Invalid resizer {resizer} specified")
    elif resizer == "clean":
        return make_resizer("PIL", "bicubic", (size, size))
    elif resizer == "legacy":
        return make_resizer("PyTorch", "bilinear", (size, size))

def make_resizer(library, filter, output_size):
    if library == "PIL":
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img).reshape(s1, s2, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func



def quantize_images(x):
    x = (x+1)/2
    x = (255.0*x + 0.5).clamp(0.0, 255.0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x

def resize_images(x, resizer, ToTensor, mean, std, device):
    x = x.transpose((0, 2, 3, 1))
    x = list(map(lambda x: ToTensor(resizer(x)), list(x)))
    x = torch.stack(x, 0).to(device)
    x = (x/255.0 - mean)/std
    return x
    