from pathlib import Path
import cv2
import os
import ray
from mtcnn import MTCNN
detector = MTCNN()

in_root = '/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq/images1024x1024'
out_detection = os.path.join('/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq','detections')

if not os.path.exists(out_detection):
    os.makedirs(out_detection)

path = Path(in_root)
img_path_list = list(path.rglob("*.png"))
gpus = 8
imgs_per_threads = len(img_path_list)//gpus


@ray.remote(num_gpus=gpus, max_calls=gpus)
def mtcnn_detection(path_list):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    # detector = MTCNN()
    for img in path_list:
        print(img)
        img_str = str(img)
        img_name = img.parts[-1]
        if img_name.endswith(".jpg"):
            dst = os.path.join(out_detection, img_name.replace(".jpg", ".txt"))
        if img_name.endswith(".png"):
            dst = os.path.join(out_detection, img_name.replace(".png", ".txt"))
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
    
@ray.remote(num_gpus=gpus, max_calls=gpus)
def test():
    print(123)
    
if __name__ == '__main__':
    num_process = gpus
    num_gpus = gpus
    ray.init(num_cpus=num_process, num_gpus=num_gpus)
    
    for i in range(num_gpus):
        print(i)
        path_list = img_path_list[imgs_per_threads * i : imgs_per_threads * (i+1)]
        if i == 7:
            path_list.extend(path_list[imgs_per_threads * (i+1):])
        # mtcnn_detection.remote(path_list)
        # test.remote()
        mtcnn_detection.remote(path_list)