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
syn_folder = '/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/syn_data'
out_folder = os.path.join(syn_folder, 'render')

syn_folder = '/home/aiteam/tykim/temp/disco'
out_folder = '/home/aiteam/tykim/temp/disco'

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
print(len(param_list))

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

    # print(coeff_init)
    
    # print(coeff_init[0] == coeff_init[1])
    
    with torch.no_grad():
        output_coeff = torch.tensor(coeff_init, dtype=torch.float32).to(model.device)
        model.facemodel.to(model.device)
        pred_vertex, pred_tex, pred_color, pred_lm = model.facemodel.compute_for_render(output_coeff)
        pred_mask, _, pred_face = model.renderer(pred_vertex, model.facemodel.face_buf, feat=pred_color)
        
    
    # print(pred_face.shape)
    # for j in range(pred_face.shape[0]):
    #     pred_face_numpy = util.tensor2im(pred_face[j])
    #     # print(pred_face_numpy.shape)
    #     padded_idx = '{:04d}'.format(j)
    #     img_path = os.path.join(out_folder,f'{int(subject_name)}_{int(padded_idx)}_r.png')
    #     util.save_image(pred_face_numpy, img_path)
    
    
    for j in range(pred_mask.shape[0]):
        pred_mask_numpy = util.tensor2im(pred_mask[j])
        img_path = os.path.join(out_folder,f'{int(subject_name)}_{int(j)}_m.png')
        util.save_image(pred_mask_numpy, img_path)
        
    break
        
# with open('test/test.npy', 'rb') as f:
#     coeff = np.load(f)

# id_coeff = coeff[:, :80]
# tex_coeff = coeff[:, 80:80+80]
# exp_coeff = coeff[:, 160:160+64]
# gamma_coeff = coeff[:, 160+64:160+64+27]
# angle_coeff = coeff[:, 160+64+27:160+64+27+3]
# # trans_coeff = np.zeros([1, 3])
# # print(trans_coeff.shape)
# # print(coeff.shape)

# coeff_init = np.zeros([1, 254+3])
# coeff_init[0, :80] = id_coeff # id layer
# coeff_init[0, 80:80+64] = exp_coeff # exp layer
# coeff_init[0, 80+64:80+64+80] = tex_coeff # tex layer
# coeff_init[0, 80+64+80:80+64+80+3] = angle_coeff # angle layer
# coeff_init[0, 80+64+80+3:80+64+80+3+27] = gamma_coeff  # gamma layer
# # coeff_init[0, 80+64+80+3+27:80+64+80+3+27+3] = trans_coeff

# print(coeff_init.shape)

# opt = TestOptions().parse() 

# rank = 0
# device = torch.device(rank)
# torch.cuda.set_device(device)

# model = create_model(opt)
# model.setup(opt)

# model.device = device
# model.parallelize()
# model.eval()


# # forward
# with torch.no_grad():
#     output_coeff = torch.tensor(coeff_init, dtype=torch.float32).to(model.device)
#     model.facemodel.to(model.device)
#     pred_vertex, pred_tex, pred_color, pred_lm = model.facemodel.compute_for_render(output_coeff)
#     pred_mask, _, pred_face = model.renderer(pred_vertex, model.facemodel.face_buf, feat=pred_color)


# print(pred_face.shape)
# pred_face_numpy = util.tensor2im(pred_face[0])
# print(pred_face_numpy.shape)
# img_path = os.path.join('/home/aiteam/tykim/generative/gan/3D-FM-GAN/models/Deep3DFaceRecon_pytorch/test','test.png')
# util.save_image(pred_face_numpy, img_path)