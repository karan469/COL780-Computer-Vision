import os
import cv2
import numpy as np

# target_folder = './train/'
# count = 1
# # for filename in os.listdir('./train'):
# #     img = cv2.imread(target_folder + filename)
# #     h, w, d = img.shape
# #     new = cv2.resize(img, (50, 50))
# #     os.system('rm ' + target_folder + filename)
# #     print('Deleted Previous ==> ' + str(count))
# #     cv2.imwrite('./train/' + filename, new)
# #     print('Written resized image: ' + filename)
# #     count += 1

# dest_for_canny_train = './canny/train/'
# count2 = 1
# for filename in os.listdir('./train'):
# 	img = cv2.imread(target_folder + filename, 0)
# 	edges = cv2.Canny(img, 100, 200)
# 	cv2.imwrite(dest_for_canny_train + filename, edges)
# 	print('Written file => ' + dest_for_canny_train + filename)
# 	count2 += 1

target_folder = './test1/'
dest_for_canny_train = './canny/test1/'
count2 = 1
for filename in os.listdir('./test1'):
	img = cv2.imread(target_folder + filename, 0)
	new = cv2.resize(img, (50, 50))
	edges = cv2.Canny(new, 100, 200)
	# print(edges)
	cv2.imwrite(dest_for_canny_train + filename, edges)
	print('Written file => ' + dest_for_canny_train + filename)
	count2 += 1
