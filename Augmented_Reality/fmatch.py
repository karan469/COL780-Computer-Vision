import numpy as np
import cv2

def projection_matrix(camera_parameters, homography):
	"""
	 From the camera calibration matrix and the estimated homography
	 compute the 3D projection matrix
	 """
	# Compute rotation along the x and y axis as well as the translation
	homography = homography * (-1)
	rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]
	# normalise vectors
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	translation = col_3 / l
	# compute the orthonormal basis
	c = rot_1 + rot_2
	p = np.cross(rot_1, rot_2)
	d = np.cross(c, p)
	rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_3 = np.cross(rot_1, rot_2)
	# finally, compute the 3D projection matrix from the model to the current frame
	projection = np.stack((rot_1, rot_2, rot_3, translation)).T
	return np.dot(camera_parameters, projection)


MIN_MATCHES = 15
cap = cv2.imread('scene.jpg', 0)    
model = cv2.imread('model.jpg', 0)
# ORB keypoint detector
# orb = cv2.ORB_create()              
orb = cv2.ORB_create(nfeatures = 1000, scoreType=cv2.ORB_FAST_SCORE) # Initiate SIFT detector
orb.setPatchSize(50);	
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None)  
# Compute scene keypoints and its descriptors
kp_frame, des_frame = orb.detectAndCompute(cap, None)
# Match frame descriptors with model descriptors
matches = bf.match(des_model, des_frame)
# matches = bf.knnMatch(des_model, des_frame,k=1)
# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    # draw first 15 matches.
    cap = cv2.drawMatches(model, kp_model, cap, kp_frame,
                          matches[:MIN_MATCHES], 0, flags=2)
    # show result
    cv2.imshow('frame', cap)
    cv2.waitKey(0)
else:
    print ("Not enough matches have been found - %d/%d" % (len(matches),
                                                          MIN_MATCHES))

# assuming matches stores the matches found and 
# returned by bf.match(des_model, des_frame)
# differenciate between source points and destination points
src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# compute Homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Draw a rectangle that marks the found model in the frame
h, w = model.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# project corners into frame
dst = cv2.perspectiveTransform(pts, M)  
# connect them with lines
img2 = cv2.polylines(cap, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
cv2.imshow('frame', cap)
cv2.waitKey(0)