import numpy as np
import cv2
import glob

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
        print('true')
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)  
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)    #change parameters here
        cv2.imshow('img',img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

##############################################################################################################################

# f= open("cameraCalibration.txt","w+")
# f.write(ret)
# f.write(mtx,"\n")
# f.write(dist,"\n")
# f.write(rvecs,"\n")
# f.write(tvecs,"\n")
# f.close()

img = cv2.imread('Images/p5.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
print('optimized')
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
cv2.imshow('img',img)
cv2.waitKey(1000)

# print(dst)

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
print('draw')

axis2 = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

cap = cv2.VideoCapture('video1.webm')
# ret, frame = cap.read()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
# for fname in glob.glob('Images/p*.jpg'):
    # img = cv2.imread(frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)

        frame = drawCube(frame,corners2,imgpts)
        print('drew')
        # out.write(frame)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(5) & 0xff
        # if k == 's':
        #     cv2.imwrite(frame[:6]+'.png', frame)
        # k = cv2.waitKey(1) & 0xff
        if k == 2700:
            break
cap.release()
cv2.destroyAllWindows()