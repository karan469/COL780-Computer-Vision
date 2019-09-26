import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math

img1 = cv2.imread(sys.argv[1],0) # queryImage-1
img2 = cv2.imread(sys.argv[2],0) # queryImage-2

orb = cv2.ORB_create(nfeatures = 100000, scoreType=cv2.ORB_FAST_SCORE) # Initiate SIFT detector

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# findHomography_(img1, img2)

def diffArray(des1, des2):
	res = []
	for i in range(len(des1)):
			res.append(des1[i]-des2[i])
	return res


def squareArray(des1):
	res = 0
	for i in range(len(des1)):
		res = res + des1[i]*des1[i]
	return math.sqrt(res)

# print((diffArray(des1,des2)))
# print(np.diff(des1,des2))

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
dmatches = sorted(matches, key = lambda x:x.distance)

## extract the matched keypoints
src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

h,w = img1.shape[:2]
print(h,w)
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
cv2.imshow("found", img2)

res = cv2.drawMatches(img1, kp1, img2, kp2, dmatches[:20],None,flags=2)

cv2.imshow("orb_match", res);

# dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show()
# plt.figure()
# dst[0:img2.shape[0], 0:img2.shape[1]] = img2
# cv2.imwrite('output.jpg',dst)
# plt.imshow(dst)
# plt.show()

cv2.waitKey();cv2.destroyAllWindows()

# draw only keypoints location,not size and orientation
# img1_new = cv2.drawKeypoints(img1,kp1,img1,color=(0,255,0), flags=0)
# img2_new = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,0), flags=0)

# plt.imshow(img1_new),plt.show()
# plt.imshow(img2_new),plt.show()
