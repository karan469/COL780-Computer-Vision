import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import sys
import math

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# img1 = cv2.imread(sys.argv[1],0) # queryImage-1
# img2 = cv2.imread(sys.argv[2],0) # queryImage-2

orb = cv2.ORB_create(nfeatures = 100000, scoreType=cv2.ORB_FAST_SCORE) # Initiate SIFT detector

images = load_images_from_folder(str(input()))
index=0
kp_temp,des_temp = orb.detectAndCompute(images[index], None)
kp = [kp_temp]
des = [des_temp]
index +=1
while index<len(images):
	kp_temp,des_temp = orb.detectAndCompute(images[index], None)
	kp.append(kp_temp)
	des.append(des_temp)
	index +=1
# kp1, des1 = orb.detectAndCompute(imgages[0], None)
# kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches1 = bf.match(des[1], des[2])
# dmatches1 = sorted(matches1, key = lambda x:x.distance)
print("num of match ",len(matches1))

# sorted_images = images
images_left = [-1]*len(images)
images_right = [-1]*len(images)
print(images_left)
def find_right(index):
	if images_left[index] == (-1):
		i=0
		max_match=3000		##choose wise threshold
		max_index=-1
		j=0
		for descripter in des:
			if j!=index :
				matches_right = bf.match(des[j], des[index])
				if max_match<len(matches_right):
					max_match = len(matches_right)
					max_index = j
			j +=1
		if max_index != -1:
			if (bf.match(des[max_index], des[index])[0].distance) < (bf.match(des[index], des[max_index])[0].distance):
				images_right[index] = max_index
				images_left[max_index] = index
			else:
				images_left[index] = max_index
				images_right[max_index] = index
				max_match=3000
				max_index=-1
				j=0
				for descripter in des:
					if j!=index and j!=images_left[index]:
						matches_right = bf.match(des[j], des[index])
						if max_match<len(matches_right):
							max_match = len(matches_right)
							max_index = j
					j +=1
				if max_index != -1:
					images_right[index] = max_index
					images_left[max_index] = index
	else:
		max_match=3000
		max_index=-1
		j=0
		for descripter in des:
			if j!=index and j!=images_left[index]:
				matches_right = bf.match(des[j], des[index])
				if max_match<len(matches_right):
					max_match = len(matches_right)
					max_index = j
			j +=1
		if max_index != -1:
			images_right[index] = max_index
			images_left[max_index] = index		

def find_left(index):
	if images_right[index] == (-1):
		i=0
		max_match=3000
		max_index=-1
		j=0
		for descripter in des:
			if j!=index :
				matches_left = bf.match(des[j], des[index])
				if max_match<len(matches_left):
					max_match = len(matches_left)
					max_index = j
			j +=1
		if max_index != -1:
			if (bf.match(des[max_index], des[index])[0].distance) > (bf.match(des[index], des[max_index])[0].distance):
				images_left[index] = max_index
				images_right[max_index] = index
			else:
				images_right[index] = max_index
				images_left[max_index] = index
				max_match=3000
				max_index=-1
				j=0
				for descripter in des:
					if j!=index and j!=images_right[index]:
						matches_left = bf.match(des[j], des[index])
						if max_match<len(matches_left):
							max_match = len(matches_left)
							max_index = j
					j +=1
				if max_index != -1:
					images_left[index] = max_index
					images_right[max_index] = index
	else:
		max_match=3000
		max_index=-1
		j=0
		for descripter in des:
			if j!=index and j!=images_right[index]:
				matches_left = bf.match(des[j], des[index])
				if max_match<len(matches_left):
					max_match = len(matches_left)
					max_index = j
			j +=1
		if max_index != -1:
			images_left[index] = max_index
			images_right[max_index] = index		

### calling functions to complete image_right and image_left matrices
i = 0
for img in images:
	if images_right[i] == -1:
		find_right(i)
	if images_left[i] == -1:
		find_left(i)
	i +=1
# match_index = find_max(match_temp,index)
# get_order(index,match_index)
print(images_right)
print(images_left)

###### order images ######
i=0
start = -1
for val in images_left:
	if images_left[i] == -1:
		start = images_right[i]
		break
	i +=1
sorted_images = [images[start]]
i_next = start
while images_right[i_next]!=-1:
	# print(i_next)
	i_next = images_right[i_next]
	sorted_images.append(images[i_next])
## now we have an ordered array of images in sorted_images

# index=0
# for descripter in des:
# 	match_id = find_neigh()
# 	index +=1

# findHomography_(img1, img2)
###################################################################################################################################3
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
matches1 = bf.match(des1, des2)
dmatches1 = sorted(matches1, key = lambda x:x.distance)
min_match1 = matches1[0].distance
matches2 = bf.match(des2, des1)
dmatches2 = sorted(matches2, key = lambda x:x.distance)
min_match2 = matches2[0].distance

if min_match2 < min_match1:
	image_temp = des1
	des1 = des2
	des2 = image_temp
	kp_temp = kp1
	kp1 = kp2
	kp2 = kp_temp
	dmatches1 = dmatches2



## extract the matched keypoints
src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches1]).reshape(-1,1,2)
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches1]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

print(M)

h,w = img1.shape[:2]
# print(h,w)
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

# print(pts)

img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
# cv2.imshow("found", img2)

res = cv2.drawMatches(img1, kp1, img2, kp2, dmatches1[:20],None,flags=2)

# cv2.imshow("orb_match", res);

dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show() # until this img2 is warped and ready to be stitched
# plt.figure()
dst[0:img2.shape[0], 0:img2.shape[1]] = img2
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()


cv2.waitKey();cv2.destroyAllWindows()

# draw only keypoints location,not size and orientation
# img1_new = cv2.drawKeypoints(img1,kp1,img1,color=(0,255,0), flags=0)
# img2_new = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,0), flags=0)

# plt.imshow(img1_new),plt.show()
# plt.imshow(img2_new),plt.show()
