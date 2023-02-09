import os
from pathlib import Path

from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import random
from PIL import Image

######### Utils ##############
def pil_rgb_convert(image):
    if not image.mode == 'RGB':
        image = image.convert("RGB")

    return image

######### Transformation ##############
class CenterCropMargin(object):
    def __init__(self, fraction=0.95):
        super().__init__()
        self.fraction=fraction
        
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size)*self.fraction)

    def __repr__(self):
        return self.__class__.__name__


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data_dict):

        if torch.rand(1) < self.p:
            return {k: TF.hflip(v) for k, v in data_dict.items()}
        return data_dict

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, data_dict):
        return {k: TF.resize(v, self.size, self.interpolation, self.max_size, self.antialias) for k, v in data_dict.items()}

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)

class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, data_dict):
        data_dict = {k: TF.normalize(v, self.mean, self.std, self.inplace) for k, v in data_dict.items()}
        return data_dict
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class ToTensor:
    def __call__(self, data_dict):
        return {k: TF.to_tensor(v) for k, v in data_dict.items()}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ReconDataset(Dataset):
    """
    Reconstruction
    recon : snythetic or ffhq
    """
    def __init__(self, root, img_range=(0, 65000), 
                 sub_dir='ffhq',
                 crop=False,
                 resize_size=None,
                 interpolation='lanczos',
                 random_flip=False,
                 normalize=True):
        
        self.root = Path(root, sub_dir)
        
        self.img_p_list = sorted(list(Path(self.root, 'images').rglob('*[!_r].png')))[img_range[0]:img_range[1]]
        self.img_r_list = sorted(list(Path(self.root, 'rendering').rglob('*_r.png')))[img_range[0]:img_range[1]]

        assert len(self.img_p_list) == len(self.img_r_list), 'the number of pic_image != the number of rnd image '


        self.interpolation = {"nearest": InterpolationMode.NEAREST,
                              "box": InterpolationMode.BOX,
                              "bilinear": InterpolationMode.BILINEAR,
                              "bicubic": InterpolationMode.BICUBIC,
                              "lanczos": InterpolationMode.LANCZOS}[interpolation]

        self.trsf_list = []

        if crop:
            self.crop = CenterCropMargin(fraction=0.95)
        else:
            self.crop = None    

        if resize_size is not None and interpolation != 'wo_resize':
            self.resizer = transforms.Resize(resize_size, interpolation=self.interpolation)
        else:
            self.resizer = None

        if normalize:
            self.normalizer = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            self.normalizer = None

        self.to_tensor = transforms.ToTensor()
        self.random_flip = random_flip
        self.trsf = transforms.Compose(self.trsf_list)

    def transformation(self, p_img, r_img):
        if self.crop is not None:
            p_img, r_img = self.crop(p_img), self.crop(r_img)

        if self.resizer is not None:
            p_img, r_img = self.resizer(p_img), self.resizer(r_img)

        if self.random_flip and random.random() > 0.5:
            p_img = TF.hflip(p_img)
            r_img = TF.hflip(r_img)

        p_img, r_img = self.to_tensor(p_img), self.to_tensor(r_img)

        if self.normalizer is not None:
            p_img, r_img = self.normalizer(p_img), self.normalizer(r_img)
        return p_img, r_img

    def __len__(self):
        return len(self.img_p_list)

    def __getitem__(self, idx):
        p_image = pil_rgb_convert(Image.open(self.img_p_list[idx]))
        r_image = pil_rgb_convert(Image.open(self.img_r_list[idx]))
        p_image, r_image = self.transformation(p_image, r_image)
        
        return {'p1': p_image, 'r1':r_image}
        
from itertools import combinations

class DisentangleDataset(Dataset):
    def __init__(self, root, sub_dir='syn_data',
                 num_variations=7,
                 crop=False,
                 resize_size=None,
                 interpolation='lanczos',
                 random_flip=False,
                 normalize=True):
        
        self.path = Path(root, sub_dir)
        self.image_path = self.path.joinpath('images')
        self.rendering_path = self.path.joinpath('rendering')
        self.mask_path = self.path.joinpath('mask')
        
        self.img_p_list = list(self.image_path.rglob('*[!_r].png'))
        self.num_variations = num_variations
        self.combi = list(combinations([i for i in range(num_variations)], 2))
        self.data_indicies = []

        # file_name
        num_total_id_imgs = len(self.img_p_list)//self.num_variations
        for i in range(num_total_id_imgs):
            subject_idx = '%05d'%(i)
            for j in self.combi:
                self.data_indicies.append([subject_idx, j[0], j[1]])


        self.interpolation = {"nearest": InterpolationMode.NEAREST,
                              "box": InterpolationMode.BOX,
                              "bilinear": InterpolationMode.BILINEAR,
                              "bicubic": InterpolationMode.BICUBIC,
                              "lanczos": InterpolationMode.LANCZOS}[interpolation]

        self.random_flip = random_flip
        self.trsf_list = []

        if crop:
            self.crop = CenterCropMargin(fraction=0.95)
            self.trsf_list.append(self.crop)
        else:
            self.crop = None    
           
        if resize_size is not None and interpolation != 'wo_resize':
            self.resizer = Resize(resize_size, interpolation=self.interpolation)
            self.trsf_list.append(self.resizer)
        else:
            self.resizer = None

        if random_flip > 0:
            self.flipper = RandomHorizontalFlip(random_flip)
            self.trsf_list.append(self.flipper)
        else:
            self.flipper = None

        if normalize:
            self.normalizer = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            self.normalizer = None
            
        self.to_tensor = ToTensor()
        self.trsf_list.append(self.to_tensor)
        
        self.trsf = transforms.Compose(self.trsf_list)

    def transformation(self, p_img, r_img, m_img):
        data = {'p_img' : p_img, 'r_img' : r_img, 'm_img' : m_img}
        return self.trsf(data)
            
    def __len__(self):
        return len(self.data_indicies)

    def __getitem__(self, idx):
        subject_idx, first_idx, second_idx = self.data_indicies[idx]

        p1_name = f'{subject_idx}_{first_idx}.png'
        r1_name = f'{subject_idx}_{first_idx}_r.png'
        m1_name = f'{subject_idx}_{first_idx}_m.png'
        
        p2_name = f'{subject_idx}_{second_idx}.png'
        r2_name = f'{subject_idx}_{second_idx}_r.png'
        m2_name = f'{subject_idx}_{second_idx}_m.png'

        p1_path = self.image_path.joinpath(p1_name)
        r1_path = self.rendering_path.joinpath(r1_name)
        m1_path = self.mask_path .joinpath(m1_name)

        p2_path = self.image_path.joinpath(p2_name)
        r2_path = self.rendering_path.joinpath(r2_name)
        m2_path = self.mask_path.joinpath(m2_name)
        
        p1 = pil_rgb_convert(Image.open(p1_path))
        r1 = pil_rgb_convert(Image.open(r1_path))
        m1 = pil_rgb_convert(Image.open(m1_path))
        
        p2 = pil_rgb_convert(Image.open(p2_path))
        r2 = pil_rgb_convert(Image.open(r2_path))
        m2 = pil_rgb_convert(Image.open(m2_path))

        return (self.transformation(p1,r1,m1), self.transformation(p2,r2,m2))

###########################
# Infinite Iterator Wrapper
###########################
class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        """The number of current epoch.
        Returns:
            int: Epoch number.
        """
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            print('stop iteration')
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

from torch.utils.data import DistributedSampler
import itertools
def sample_data(loader):
    for _epoch in itertools.count(1):
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(_epoch)
        for batch in loader:
            yield batch


class ImageDataset(Dataset):
    def __init__(self, root, sub_dir='ffhq', 
                 resize_size=None, 
                 interpolation='lanczos', 
                 img_range=[65_000, 70_000], normalize=True):
        self.root = Path(root, sub_dir)
        self.img_list = sorted(list(Path(self.root, 'images').rglob('*[!_r].png')))[img_range[0]:img_range[1]]

        self.normalize = normalize
        self.interpolation = {"nearest": InterpolationMode.NEAREST,
                              "box": InterpolationMode.BOX,
                              "bilinear": InterpolationMode.BILINEAR,
                              "bicubic": InterpolationMode.BICUBIC,
                              "lanczos": InterpolationMode.LANCZOS}[interpolation]


        self.trsf_list = []

        if resize_size is not None and interpolation != 'wo_resize':
            self.resizer = Resize(resize_size, interpolation=self.interpolation)
            self.trsf_list.append(self.resizer)

        if self.normalize:
            self.trsf_list += [transforms.ToTensor()]
            self.trsf_list += [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

        else:
            self.trsf_list += [transforms.PILToTensor()]

        self.trsf = transforms.Compose(self.trsf_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_pil = Image.open(self.img_list[idx]).convert('RGB')
        return self.trsf(img_pil)


# import os
# from pathlib import Path
# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

# class SynDataset(Dataset):
#     def __init__(self, folder):
#         self.folder = folder
#         self.path = Path(folder)

#         self.subject_list = list(path'.glob('*_*.csv'))
#     def __len__(self):
#         return len(self.subject_list)

#     def __getitem__(self, idx):
#         # os.path.join(f'{idx}_{}.jpg')
#         self.path.joinpath(f'{idx}_{}.jpg')
#         img = cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
#         img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)