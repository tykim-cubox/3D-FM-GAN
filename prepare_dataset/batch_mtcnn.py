from mtcnn import MTCNN
import random
from pathlib import Path, PosixPath
import cv2
import os
detector = MTCNN()

# python batch_mtcnn.py --in_root /home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq/images1024x1024
in_root = '/workspace/gan/3D-FM-GAN/data/ffhq/images'
out_detection = os.path.join('/workspace/gan/3D-FM-GAN/data/ffhq','final_detections')

if not os.path.exists(out_detection):
    os.makedirs(out_detection)

path = Path(in_root)
img_path_list = list(path.rglob("*.png"))
img_path_list = [PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/17000/17329.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/33000/33594.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/55000/55371.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/58000/58221.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/22000/22194.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/04000/04779.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/68000/68241.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/04000/04229.png'),
 PosixPath('/workspace/gan/3D-FM-GAN/data/ffhq/images/27000/27566.png')]


for img in img_path_list:
    print(img)
    img_str = str(img)
    img_name = img.stem

    dst = os.path.join(out_detection, f'{img_name}.txt')
    # if img_name.endswith(".jpg"):
    #     dst = os.path.join(out_detection, img_name.replace(".jpg", ".txt"))
    # if img_name.endswith(".png"):
    #     dst = os.path.join(out_detection, img_name.replace(".png", ".txt"))
    if not os.path.exists(dst):
        print(dst)
        image = cv2.cvtColor(cv2.imread(img_str), cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)

        if len(result) > 0:
            index = 0
            if len(result)>1:
                size = -100000
                for r in range(len(result)):
                    size_ = result[r]["box"][2] + result[r]["box"][3]
                    if size < size_:
                        size = size_
                        index = r

            bounding_box = result[index]['box']
            keypoints = result[index]['keypoints']
            if result[index]["confidence"] > 0.9:

                # if img_str.endswith(".jpg"):
                #     dst = os.path.join(out_detection, img_str.replace(".jpg", ".txt"))
                # if img_str.endswith(".png"):
                #     dst = os.path.join(out_detection, img_str.replace(".png", ".txt"))

                outLand = open(dst, "w")
                outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['nose'][0])) + " " +      str(float(keypoints['nose'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
                outLand.close()
                print(result)   