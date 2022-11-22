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

variations = 7
opt = TestOptions().parse()


syn_folder = '/home/aiteam/tykim/temp/disco'
out_folder = os.path.join('/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/syn_data', 'mask')

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    
def load_param_list(folder=syn_folder):
    path = Path(folder)    
    param_list = list(path.glob('*.npy'))
    return param_list


def load_model():
    rank = 0
    device = torch.device(rank)
    torch.cuda.set_device(device)

    model = create_model(opt)
    model.setup(opt)

    model.device = device
    model.parallelize()
    model.eval()
    return model
  
model = load_model()
param_list = load_param_list()

# 10_000개에 대해서 
for i in range(len(param_list)):
    with open(param_list[i], 'rb') as f:
        coeff = np.load(f)
        
    subject_name = param_list[i].parts[-1].replace('.npy', '')
    print(subject_name, coeff.shape)
    
    id_coeff = coeff[:, :80]
    tex_coeff = coeff[:, 80:80+80]
    exp_coeff = coeff[:, 160:160+64]
    angle_coeff = coeff[:, 160+64:160+64+3]
    gamma_coeff = coeff[:, 160+64+3:160+64+3+27]
    
    coeff_init = np.zeros([variations, 254+3])
    
    coeff_init[:, :80] = id_coeff # id layer
    coeff_init[:, 80:80+64] = exp_coeff # exp layer
    coeff_init[:, 80+64:80+64+80] = tex_coeff # tex layer
    coeff_init[:, 80+64+80:80+64+80+3] = angle_coeff # angle layer
    coeff_init[:, 80+64+80+3:80+64+80+3+27] = gamma_coeff  # gamma layer
    
    
    with torch.no_grad():
        output_coeff = torch.tensor(coeff_init, dtype=torch.float32).to(model.device)
        model.facemodel.to(model.device)
        pred_vertex, pred_tex, pred_color, pred_lm = model.facemodel.compute_for_render(output_coeff)
        pred_mask, _, pred_face = model.renderer(pred_vertex, model.facemodel.face_buf, feat=pred_color)
        
    for j in range(pred_mask.shape[0]):
        pred_mask_numpy = util.tensor2im(pred_mask[j])
        img_path = os.path.join(out_folder,f'{int(subject_name)}_{int(j)}_m.png')
        util.save_image(pred_mask_numpy, img_path)