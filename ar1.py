import numpy as np
import cv2
import glob
import os
from obj_loader import *
from OpenGL.GL import *
import pygame
import argparse
import math

MIN_MATCHES = 10

parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')

args = parser.parse_args()

def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 1
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Images/p*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)			#change parameters here
    # print('ytueb')

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # print('true')
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)  
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)    #change parameters here
        cv2.imshow('img',img)
        # cv2.waitKey(500)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread('Images/p5.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
cv2.imshow('img',img)
cv2.waitKey(1000)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)


def drawLine(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

axis1 = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
# print('draw')

axis2 = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

#-----------------------------RENDER FUNC----------------------------------------------
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
#-------------------------------------------------------------------------------------

homography = None 
orb = cv2.ORB_create(nfeatures = 1000000, scoreType=cv2.ORB_FAST_SCORE) # Initiate SIFT detector
orb.setPatchSize(80);
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
dir_name = os.getcwd()
model = cv2.imread(os.path.join(dir_name, 'model.png'), 0)
kp_model, des_model = orb.detectAndCompute(model, None)
obj = OBJ(os.path.join(dir_name, 'models/wolf.obj'), swapyz=True)

#-----------------------FRAME READ---------------------------------------------
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    frame = dst[y:y+h, x:x+w]
    # cv2.imwrite('calibresult.png',frame)

    #---------------------------WITHOUT CALIB FUNC-----------------------------------------------
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    # print(kp_frame[0].pt)
    # match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)
    # print(matches)
    matches = sorted(matches, key=lambda x: x.distance)

    # compute Homography if enough matches are found
    if len(matches) > MIN_MATCHES:
        # differenciate between source points and destination points
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if args.rectangle:
            h, w = model.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, homography)
            # connect them with lines  
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
        # if a valid homography matrix was found render cube on model plane
        if homography is not None:
            print("homography not none")
            try:
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(mtx, homography)  
                print(projection)

                frame = render(frame, obj, projection, model, False)
                #frame = render(frame, model, projection)
            except:
                print("B")
                pass
        # draw first 10 matches.
        if args.matches:
            frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
        # show result
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print "Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES)
    #--------------------------------------------------------------------------

    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
    # if ret == True:
    #     corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    #     # Find the rotation and translation vectors.
    #     _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

    #     # project 3D points to image plane
    #     imgpts, jac = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)

    #     frame = drawCube(frame,corners2,imgpts)
    #     print('drew')
    #     # out.write(frame)
    #     cv2.imshow('frame',frame)
    #     k = cv2.waitKey(5) & 0xff
    #     # if k == 's':
    #     #     cv2.imwrite(frame[:6]+'.png', frame)
    #     # k = cv2.waitKey(1) & 0xff
    #     if k == 2700:
    #         break
cap.release()
cv2.destroyAllWindows()


