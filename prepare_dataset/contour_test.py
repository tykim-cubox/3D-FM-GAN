import cv2
import numpy as np
from pathlib import Path

r_img_list = list(Path('/home/aiteam/tykim/temp/disco').rglob('*_r.png'))


# Read image
im_in = cv2.imread(str(r_img_list[0]), cv2.IMREAD_GRAYSCALE)
 
# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.|
 
th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_out = im_th | im_floodfill_inv

# Display images.
cv2.imshow("Thresholded Image", im_th)
cv2.imshow("Floodfilled Image", im_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()