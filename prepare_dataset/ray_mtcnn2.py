import ray
import time
import cv2
from pathlib import Path
import os
import sys
import psutil
from mtcnn import MTCNN


@ray.remote#(num_cpus=32, num_gpus=8)
class Model(object):
    def __init__(self, i, out_detection):
        if sys.platform == 'linux':
            psutil.Process().cpu_affinity([i])
        self.detector = MTCNN()
        self.out_detection = out_detection

    def detection(self, img_path):
        img_str = str(img_path)
        img_name = img_path.parts[-1]
        # print(img_name)
        if img_name.endswith(".jpg"):
            dst = os.path.join(self.out_detection, img_name.replace(".jpg", ".txt"))
        if img_name.endswith(".png"):
            dst = os.path.join(self.out_detection, img_name.replace(".png", ".txt"))

        if not os.path.exists(dst):
            # print(dst)
            image = cv2.cvtColor(cv2.imread(img_str), cv2.COLOR_BGR2RGB)
            result = self.detector.detect_faces(image)

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
                    outLand = open(dst, "w")
                    outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                    outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                    outLand.write(str(float(keypoints['nose'][0])) + " " +      str(float(keypoints['nose'][1])) + "\n")
                    outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                    outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
                    outLand.close()
                    # print(result)
        return dst
            

        
if __name__ == '__main__':
    num_processes = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_processes, num_gpus=8)

    in_root = '/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq/images'
    out_detection = os.path.join('/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq','detections')

    if not os.path.exists(out_detection):
        os.makedirs(out_detection)
    path = Path(in_root)
    img_path_list = list(path.rglob("*.png"))
    
    img_path_list = sorted(img_path_list)
    
    # txt_list = list(Path('/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq/detections').rglob('*.txt'))
    
    # for i in txt_list:
    #     idx = i.parts[-1].replace(".txt", "")
    #     print(idx)
    #     # img_path_list.pop(int(idx))
    #     for j in img_path_list:
    #         if idx in j

    print(len(img_path_list))

    actors = [Model.remote(i, out_detection) for i in range(num_processes)]

    start_time = time.time()
    for i, image_path in enumerate(img_path_list):
        print(i)
        actors[i%num_processes].detection.remote(image_path)
        # if i%num_processes == 0:
        #     time.sleep(2)
    # results = [actors[i%num_processes].detection.remote(image_path) for i, image_path in enumerate(img_path_list)]
    elapsed_time = time.time() - start_time
    print(elapsed_time)