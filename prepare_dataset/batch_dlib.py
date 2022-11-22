
from tqdm import tqdm
import os
import numpy as np
import argparse
import dlib
import cv2
from glob import glob
from pathlib import Path
import ray

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



def save_lmks_to_file(lmks, file):
    with open(file, "w") as f:
        for i in range(len(lmks)):
            f.write(str(lmks[i][0])+" "+str(lmks[i][0]))  # x,y coordiantes

            if i != len(lmks)-1:
                f.write("\n")
    

# +--------------------------------------------------------------------------     
#  Extract facial landmarks (68 points and 5 points are both supported).
#
#  Running example:
#
#       python lmk_dlib_detect.py -i <XXX.png or path_to_image_dir> \
#                                    -o <output_dir> \
#                                   [-p dlib_shape_predictor] \
#                                   [-v vis_dir] \
#                                   [-lmk5p vis_dir]
#                                   
# +--------------------------------------------------------------------------       

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat", help="path to facial landmark predictor")
# ap.add_argument("-i", "--input", type=str, default= "./realign1500/", help="path to input image or input folders")
# ap.add_argument("-o", "--output", type=str, default= "lmk", help="path to output directory for landmark saving")
# ap.add_argument("-v", "--vis", type=str, default="", help="path to output directory for landmark visualization")
# ap.add_argument("--lmk5p", action="store_true", default=True, help="convert lmk68 to lmk5")
# args = vars(ap.parse_args())


shape_predictor_path = '/home/aiteam/tykim/generative/gan/3D-FM-GAN/prepare_dataset/shape_predictor_68_face_landmarks.dat'
in_root = '/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq/images1024x1024'
out_detection = os.path.join('/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq','detections2')


# if os.path.isdir(args["input"]):
#     imgs = glob(os.path.join(args["input"], "*.*g"))
# else:
#     imgs = [args["input"]]

# os.makedirs(args["output"], exist_ok=True)
# if args["vis"]!="":
#     os.makedirs(args["vis"], exist_ok=True) 

if not os.path.exists(out_detection):
    os.makedirs(out_detection)

path = Path(in_root)
img_path_list = list(path.rglob("*.png"))
gpus = 8
imgs_per_threads = len(img_path_list)//gpus


@ray.remote
def mtcnn_detection(path_list, detector, predictor):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print('dsads')
    for img in tqdm(path_list, total=len(path_list)):
        print(img)
        img_str = str(img)
        img_name = img.parts[-1]
        if img_name.endswith(".jpg"):
            dst = os.path.join(out_detection, img_name.replace(".jpg", ".txt"))
        if img_name.endswith(".png"):
            dst = os.path.join(out_detection, img_name.replace(".png", ".txt"))
        if not os.path.exists(dst):
            print(dst)
            image = cv2.imread(img_str)
            H, W = image.shape[0], image.shape[1]
            rects = detector(image, 1)

            lmks = None
            face_cnt = 0

            # loop over the face detections
            for (i, rect) in enumerate(rects):
                # Only extract one face per image
                if face_cnt >= 1: break

                # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = predictor(image, rect)
                lmks = lmk_to_np(shape)

                if np.sum(lmks < 0) > 0 or np.sum(lmks >= H) > 0 or np.sum(lmks >= W) > 0: 
                    continue
                else:
                    face_cnt += 1
                    
                lmks = extract_5p(lmks)
            
            if lmks is not None:
                save_lmks_to_file(lmks, dst)
                # print("lmks: ",lmks)
                
                
if __name__ == '__main__':
    gpus = 8
    num_gpus = gpus
    ray.init(num_cpus=8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    
    detector_ray = ray.put(detector)
    predictor_ray = ray.put(predictor)
    
    for i in range(num_gpus):
        print(i)
        path_list = img_path_list[imgs_per_threads * i : imgs_per_threads * (i+1)]
        if i == 7:
            path_list.extend(path_list[imgs_per_threads * (i+1):])
        # mtcnn_detection.remote(path_list)
        # test.remote()
        print(len(path_list))
        mtcnn_detection.remote(path_list, detector_ray, predictor_ray)
        break