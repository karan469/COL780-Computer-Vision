import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
import os
import glob

orb = cv2.ORB_create(nfeatures = 100000, scoreType=cv2.ORB_FAST_SCORE) # Initiate SIFT detector

# img1 = cv2.imread(sys.argv[1],0) # queryImage-1
# img2 = cv2.imread(sys.argv[2],0) # queryImage-2

# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# -------------------------------------------------------------------------------
# user_input = raw_input("Enter the path of your file: ")
# assert os.path.exists(user_input), "I did not find the file at, "+str(user_input)
# inn = user_input + '*.jpg'
# allImages = glob.glob(inn)
# sorted(allImages, key = lambda x: x.split(user_input)[1].split('.jpg')[0])
allImages = ['./dir2/1.jpeg','./dir2/2.jpeg','./dir2/3.jpeg', './dir2/4.jpeg']
# print(imgreads)
imgreads = []
for singleImg in allImages:
	print(singleImg)
	imgreads.append(cv2.imread(singleImg,0))

kp = []
des = []
for img in imgreads:
	kp.append(orb.detectAndCompute(img, None)[0])
	des.append(orb.detectAndCompute(img, None)[1])

#--------------------------------------------------------------------------------	

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

allmatches = []
alldmatches = []
for i in range(0,len(des)-2):
	allmatches.append(bf.match(des[i],des[i+1]))
	alldmatches.append(sorted(bf.match(des[i],des[i+1]), key = lambda x: x.distance))

matches = bf.match(des[0], des[1])
dmatches = sorted(matches, key = lambda x:x.distance)


## extract the matched keypoints
src_pts  = np.float32([kp[0][m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kp[1][m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# print(M)

h,w = imgreads[0].shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

imgreads[1] = cv2.polylines(imgreads[1], [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)

res = cv2.drawMatches(imgreads[0], kp[0], imgreads[1], kp[1], dmatches[:20],None,flags=2)

dst = cv2.warpPerspective(imgreads[0],M,(imgreads[1].shape[1] + imgreads[0].shape[1], imgreads[1].shape[0]))
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
dst[0:imgreads[1].shape[0], 0:imgreads[1].shape[1]] = imgreads[1]
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()


cv2.waitKey();cv2.destroyAllWindows()

