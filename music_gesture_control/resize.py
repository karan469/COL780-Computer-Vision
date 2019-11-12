import os
import cv2
import numpy as np

target_folder = './train/'
count = 1
for filename in os.listdir('./train'):
    img = cv2.imread(target_folder + filename)
    h, w, d = img.shape
    new = cv2.resize(img, (50, 50))
    os.system('rm ' + target_folder + filename)
    print('Deleted Previous ==> ' + str(count))
    cv2.imwrite('./train/' + filename, new)
    print('Written resized image: ' + filename)
    count += 1
