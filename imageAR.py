import numpy as np
import cv2
import glob
import os
from obj_loader import *
from OpenGL.GL import *
import pygame
import argparse
import math

MIN_MATCHES = 12

parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')

args = parser.parse_args()

def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.02
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
            cv2.fillConvexPoly(img, imgpts, (191,255,0))
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
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)          #change parameters here
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
# cv2.imshow('img',img)
# cv2.waitKey(1000)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

axis1 = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
# print('draw')

axis2 = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

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
    # print(translation)
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    # return np.dot(camera_parameters, projection)
    return camera_parameters, projection
#-------------------------------------------------------------------------------------

frame = cv2.imread('Part4/marker3.png')
# ret, frame = cap.read()
file = 'ping_pong_static.avi'
out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (591,219))


homography = None 
orb = cv2.ORB_create(nfeatures = 100000, scoreType=cv2.ORB_FAST_SCORE) # Initiate SIFT detector
orb.setPatchSize(80);
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
dir_name = os.getcwd()
model1 = cv2.imread(os.path.join(dir_name, 'Part4/img1.png'), 0)
model2 = cv2.imread(os.path.join(dir_name, 'Part4/img2.png'), 0)

kp_model1, des_model1 = orb.detectAndCompute(model1, None)
kp_model2, des_model2 = orb.detectAndCompute(model2, None)

obj = OBJ(os.path.join(dir_name, 'models/pokemon.obj'), swapyz=True)

#-----------------------FRAME READ---------------------------------------------
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break

dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
frame = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',frame)

#---------------------------WITHOUT CALIB FUNC-----------------------------------------------
kp_frame1, des_frame1 = orb.detectAndCompute(frame, None)
kp_frame2, des_frame2 = orb.detectAndCompute(frame, None)

# print(kp_frame[0].pt)
# match frame descriptors with model descriptors
matches1 = bf.match(des_model1, des_frame1)
matches2 = bf.match(des_model2, des_frame2)

# print(matches)
matches1 = sorted(matches1, key=lambda x: x.distance)
matches2 = sorted(matches2, key=lambda x: x.distance)

# compute Homography if enough matches are found
if (len(matches1) > MIN_MATCHES) and (len(matches2) > MIN_MATCHES):
    # differenciate between source points and destination points
    src_pts1 = np.float32([kp_model1[m.queryIdx].pt for m in matches1]).reshape(-1, 1, 2)
    dst_pts1 = np.float32([kp_frame1[m.trainIdx].pt for m in matches1]).reshape(-1, 1, 2)
    # compute Homography
    homography1, mask1 = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)

    src_pts2 = np.float32([kp_model2[m.queryIdx].pt for m in matches2]).reshape(-1, 1, 2)
    dst_pts2 = np.float32([kp_frame2[m.trainIdx].pt for m in matches2]).reshape(-1, 1, 2)
    # compute Homograph
    homography2, mask2 = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
    # if args.rectangle:
    #     h, w = model1.shape
    #     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #     # project corners into frame
    #     dst1 = cv2.perspectiveTransform(pts, homography1)
    #     # connect them with lines  
    #     frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
    # if a valid homography matrix was found render cube on model plane
        # img2 = cv2.polylines(frame, [np.int32(dst1)], True, 255, 3, cv2.LINE_AA) 
    count = 0
    toRight = True
    toLeft = False
    flag = True
    frame_copy = None
    while(True):
        frame_copy = cv2.imread('Part4/marker4.jpg')
        if (homography1 is not None) and (homography2 is not None):
            if count==50 or flag==False:
                count = 0
                flag = True
                if toRight:
                    toLeft = True
                    toRight = False
                elif toLeft :
                    toRight = True
                    toLeft = False
            print("homography1 not none ",count)
            try:
                # obtain 3D projection matrix from homography matrix and camera parameters
                cam_para1, projection1 = projection_matrix(mtx, homography1)
                cam_para2, projection2 = projection_matrix(mtx, homography2)
                print(projection1)
                print('\n')
                print(projection2)
                print('\n')
                for i in range(3):
                    # vect = []
                    for j in range(4):
                        if j==3:
                            if toRight:
                                projection1[i][j] = (projection1[i][3] + ((projection2[i][3] - projection1[i][3])/50)*count)
                            else:
                                projection1[i][j] = (projection2[i][3] - ((projection2[i][3] - projection1[i][3])/50)*count)
                        else:
                            projection1[i][j] = (projection1[i][j])
                
                projection = np.dot(cam_para1, projection1)
                print('\n')


                # print(projection1[0][3],"   ", projection2[0][3])

                frame_copy = render(frame_copy, obj, projection, model1, False)
                count +=1
                # frame_copy = render(frame_copy, model, projection)
            except:
                print("B")
                pass
        # draw first 10 matches.
        # if args.matches:
        #     frame_copy = cv2.drawMatches(model1, kp_model1, frame_copy, kp_frame1, matches1[:20], 0, flags=2)
            # frame = cv2.drawMatches(model2, kp_model2, frame, kp_frame2, matches2[:10], 0, flags=2)
        
        # frame_copy = cv2.drawMatches(model1, kp_model1, frame_copy, kp_frame1,
        #               matches1[:MIN_MATCHES], 0, flags=2)
        # frame_copy = cv2.drawMatches(model2, kp_model2, frame_copy, kp_frame2,
        #               matches2[:MIN_MATCHES], 0, flags=2)
        # show result
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        # show result
        out.write(frame_copy)
        cv2.imshow('frame', frame_copy)
        cv2.waitKey(10)
        frame_copy = None
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

else:
    print "Not enough matches found - %d/%d" % (len(matches1), MIN_MATCHES)
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
# cap.release()
out.release()
cv2.destroyAllWindows()


