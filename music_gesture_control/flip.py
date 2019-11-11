import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# fgbg = cv2.createBackgroundSubtractorMOG2() 
# count = 3195
# kernel = np.ones((3,3),np.uint8)/9
count = 0
for filename in os.listdir('./binImages/next'):
	img = cv2.imread('./binImages/next/' + filename)
	# count += 1
	img_flip_lr = cv2.flip(img, 1)
	# print(img)
	cv2.imwrite("./binImages/prev/prev." + filename.split('.')[1] + ".png", img_flip_lr)
	print('DONE ->' + './binImages/prev/prev.' + filename.split('.')[1])

# print(count)
# img_flip_lr = cv2.flip(img, 1)
# cv2.imwrite('data/dst/lena_cv_flip_lr.jpg', img_flip_lr)
