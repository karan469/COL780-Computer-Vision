import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import shutil

# fgbg = cv2.createBackgroundSubtractorMOG2() 
# count = 3195
# kernel = np.ones((3,3),np.uint8)/9
curr = './binImages/prev/'
dest = './test1/'
count = 0
cnt = 0
arr1 = os.listdir(curr)
random.shuffle(arr1)
# print(os.listdir('./binImages/next'))
for filename in arr1:
	# print(filename)
	if count < len(arr1)/5:
		os.system('mv ' + curr + filename + " " + dest + filename)
		print('COUNT ===> ' + str(cnt))
		cnt += 1
		# print('mv ' + curr + filename + " " + dest + filename)
	count += 1
	# print('COUNT ===> ' + str(count))