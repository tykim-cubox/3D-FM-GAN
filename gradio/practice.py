import gradio as gr
import torch
import sys
import os
sys.path.append('/workspace/gan/3D-FM-GAN')
# os.chdir('..')
# sys.path.append('/home/aiteam/tykim/generative/gan/3D-FM-GAN/model/Deep3DFaceRecon_pytorch')
sys.path.append('/workspace/gan/3D-FM-GAN/model/Deep3DFaceRecon_pytorch')
from model.Deep3DFaceRecon_pytorch.models import create_model
from model.Deep3DFaceRecon_pytorch.options.test_options import TestOptions
import dlib


import numpy as np


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def lmk_to_np(shape, dtype="int32"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

# lmk-68p to lmk-5p
def extract_5p(lm, dtype="int32"):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :].astype(dtype)
    
    return lm5p  # [left_eye, right_eye, nose, left_mouth, right_mouth]

dlib_path = '/workspace/gan/3D-FM-GAN/pretraiend/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)



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

def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

opt = TestOptions().parse()
# lm3d_std = load_lm3d(opt.bfm_folder)
actors = Renderer(opt)

def predict():
    ...

# with gr.Blocks() as demo:
#     gr.Markdown("Rendering test")
#     with gr.Tab("Real Image"):
#         image_input = gr.Image()
#         image_output = gr.Image()
#     image_button = gr.Button('Do Rendering')
        
#     image_button.click(predict, inputs=image_input, outputs=image_output)
demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)
demo.launch()


# import numpy as np
# import gradio as gr

# def sepia(input_img):
#     sepia_filter = np.array([
#         [0.393, 0.769, 0.189], 
#         [0.349, 0.686, 0.168], 
#         [0.272, 0.534, 0.131]
#     ])
#     sepia_img = input_img.dot(sepia_filter.T)
#     sepia_img /= sepia_img.max()
#     return sepia_img

# demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
# demo.launch()