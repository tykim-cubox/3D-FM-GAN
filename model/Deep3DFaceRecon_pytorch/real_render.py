import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from util import util
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
from pathlib import Path



from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms


class Dataset(Dataset):
    """
    test.py를 참조해서 만들면 될 것 같음
    """
    def __init__(self, img_path_list, lm_path_list, bfm_folder):
        super().__init__()

        self.img_path_list = img_path_list
        self.lm_path_list = lm_path_list
        self.lm3d_std = load_lm3d(bfm_folder)

    def __len__(self):
        return len(self.img_path_list)

    def read_data(self, im_path, lm_path, to_tensor=True):
        # to RGB 
        im = Image.open(im_path).convert('RGB')
        W,H = im.size
        lm = np.loadtxt(lm_path).astype(np.float32)
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]
        _, im, lm, _ = align_img(im, lm, self.lm3d_std)
        if to_tensor:
            # im = im.resize([224, 224])
            im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1)# .unsqueeze(0)
            lm = torch.tensor(lm)# .unsqueeze(0)
        return im, lm
    
    def __getitem__(self, idx):
        data = {}
        img, lm = self.read_data(self.img_path_list[idx], self.lm_path_list[idx], True)
        data['img'] = img
        data['lm'] = lm
        # data['img_path'] = self.img_path_list[idx]
        return data


class Renderer(object):
    def __init__(self, opt):
        rank = 0
        device = torch.device(rank)
        model = create_model(opt)
        model.setup(opt)
        model.device = device
        model.parallelize()
        model.eval()
        
        self.model = model

    def setup(self, input):
        self.model.input_img = input['imgs'].to(self.model.device) 
        self.model.atten_mask = input['msks'].to(self.model.device) if 'msks' in input else None
        self.model.gt_lm = input['lms'].to(self.model.device)  if 'lms' in input else None
        self.model.trans_m = input['M'].to(self.model.device) if 'M' in input else None
        self.model.image_paths = input['im_paths'] if 'im_paths' in input else None

    @torch.no_grad()
    def forward(self, img):
        img = img.to(self.model.device) 
        # coeff 예측
        output_coeff = self.model.net_recon(img)
        # print(output_coeff.shape)
        self.model.facemodel.to(self.model.device)
        
        self.model.pred_vertex, self.model.pred_tex, self.model.pred_color, self.model.pred_lm = self.model.facemodel.compute_for_render(output_coeff)
        self.model.pred_mask, _, self.model.pred_face = self.model.renderer.forward(self.model.pred_vertex, self.model.facemodel.face_buf, feat=self.model.pred_color)
        
        pred_coeffs_dict = self.model.facemodel.split_coeff(output_coeff)
        
        self.model.compute_visuals()
        return self.model.pred_face, self.model.pred_mask


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        # im = im.resize([224, 224])
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


opt = TestOptions().parse()

real_folder = '/workspace/gan/3D-FM-GAN/data/ffhq/images_1024'
lmd_folder = '/workspace/gan/3D-FM-GAN/data/ffhq/merge_detections'
rendering_out_folder = '/workspace/gan/3D-FM-GAN/data/ffhq/rendering'
crop_img_folder = '/workspace/gan/3D-FM-GAN/data/ffhq/images'

lm3d_std = load_lm3d(opt.bfm_folder)


actors = Renderer(opt)

img_path_list = list(Path(real_folder).rglob('*.png'))
img_path_list = sorted(img_path_list)

lm_path_list = list(Path(lmd_folder).rglob('*.txt'))
lm_path_list = sorted(lm_path_list)


# sequential processing
# i = 0
# for img_path, lm_path in zip(img_path_list, lm_path_list):
#     im_tensor, lm_tensor = read_data(img_path, lm_path, lm3d_std)
#     actors.setup({'imgs': im_tensor, 'lms': lm_tensor})
#     pred_face, pred_mask = actors.forward(im_tensor)


#     pred_face_numpy = util.tensor2im(pred_face[0])
#     util.save_image(pred_face_numpy, os.path.join(out_folder,f'{img_path.stem}_r.png'))

#     im_numpy = util.tensor2im(im_tensor[0])
#     util.save_image(im_numpy, os.path.join(out_folder,f'{img_path.stem}_align.png'))
    

#     visuals = actors.model.get_current_visuals() 
#     for idx, (label, image) in enumerate(visuals.items()):
#         print(idx)
#         print(type(label))
#         print(type(image))
#         vis_np = util.tensor2im(image[0])
#         util.save_image(vis_np, os.path.join(out_folder,f'{img_path.stem}_vis.png'))
#     i += 1
#     if i==3:
#         break


# batch processing

batch_size = 256
ds = Dataset(img_path_list, lm_path_list, opt.bfm_folder)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

for batch_idx, batch_data in enumerate(dl):
    im_tensor = batch_data['img']
    lm_tensor = batch_data['lm']
    # img_path = batch_data['img_path']
    
    print(f'{batch_idx} = ', im_tensor.shape)
    actors.setup({'imgs' : im_tensor, 'lms': lm_tensor})
    pred_face, pred_mask = actors.forward(im_tensor)

    for i in range(batch_size):
        # rendering
        pred_face_numpy = util.tensor2im(pred_face[i])
        util.save_image(pred_face_numpy, os.path.join(rendering_out_folder,f'{img_path_list[batch_idx*batch_size + i].stem}_r.png'))
        # align and cropped
        im_numpy = util.tensor2im(im_tensor[i])
        util.save_image(im_numpy, os.path.join(crop_img_folder,f'{img_path_list[batch_idx*batch_size + i].stem}.png'))



                                          

# syn_data test
# im_path = Path('/workspace/gan/2291_04.png')
# im = Image.open(im_path).convert('RGB')
# im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
# pred_face, pred_mask = actors.forward(im)

# pred_face_numpy = util.tensor2im(pred_face[0])
# util.save_image(pred_face_numpy, os.path.join(out_folder,f'{im_path.stem}_r.png'))