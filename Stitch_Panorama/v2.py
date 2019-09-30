import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
import os
import glob

def multiplyMatrix(X,Y):
	result = [[0 for x in range(3)] for y in range(3)]
	for i in range(len(X)):  
	   for j in range(len(Y[0])):  
		   for k in range(len(Y)):  
			   result[i][j] += X[i][k] * Y[k][j]
	return result

# allImages = ['./dir1/214.jpg','./dir1/213.jpg','./dir1/212.jpg']
# allImages = ['./dir2/14.jpeg','./dir2/13.jpeg','./dir2/11.jpeg']
allImages = ['./4.jpg','./3.jpg','./2.jpg', './1.jpg']

imgreads = []
for singleImg in allImages:
	print(singleImg)
	imgreads.append(cv2.imread(singleImg,0))

orb = cv2.ORB_create(nfeatures = 100000, scoreType=cv2.ORB_FAST_SCORE) # Initiate SIFT detector
kp = []
des = []
for img in imgreads:
	kp.append(orb.detectAndCompute(img, None)[0])
	des.append(orb.detectAndCompute(img, None)[1])

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
dmatchArr = []
for i in range(0, len(imgreads)-1):
	dmatchArr.append(sorted(bf.match(des[i],des[i+1]), key = lambda x: x.distance))

srcNdes = []
MNmask = []
hNw = []
ptsArr = []
# resArr = []
dstArr = []
for i in range(0, len(imgreads)-1):
	src_pts = np.float32([kp[i][m.queryIdx].pt for m in dmatchArr[i]]).reshape(-1,1,2)
	des_pts = np.float32([kp[i+1][m.trainIdx].pt for m in dmatchArr[i]]).reshape(-1,1,2)
	srcNdes.append(( src_pts, des_pts))
	
	M, mask = cv2.findHomography(src_pts, des_pts, cv2.RANSAC,5.0)
	MNmask.append((M, mask))
	
	h,w = imgreads[i].shape[:2]
	hNw.append((h,w))
	
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	ptsArr.append(pts)
	
	dst = cv2.perspectiveTransform(pts,M)
	# print(dst)
	dstArr.append(dst)
	
	imgreads[i+1] = cv2.polylines(imgreads[i+1], [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
	res1 = cv2.drawMatches(imgreads[i], kp[i], imgreads[i+1], kp[i+1], dmatchArr[i][:20],None,flags=2)

dst = []
ded =  cv2.warpPerspective(imgreads[0],MNmask[0][0],(imgreads[1].shape[1] + imgreads[0].shape[1], imgreads[1].shape[0]))
for i in range(0, len(imgreads)-1):
	if i>0:
		ded = cv2.warpPerspective(ded,MNmask[i][0],(imgreads[i+1].shape[1] + ded.shape[1], imgreads[i+1].shape[0]))
	
	ded[0:imgreads[i+1].shape[0], 0:imgreads[i+1].shape[1]] = imgreads[i+1]
	dst.append(ded)
	print(ded)

# print(len(dst))
# print(dst[0])
ded[0:imgreads[len(imgreads)-1].shape[0], 0:imgreads[len(imgreads)-1].shape[1]] = imgreads[len(imgreads)-1]
plt.subplot(122),plt.imshow(ded),plt.title('Warped Image')

cv2.imwrite('./dir1/dir1output.jpg',dst[len(dst)-1])
plt.imshow(ded)
plt.show()

cv2.waitKey();cv2.destroyAllWindows()
