
# with morphological transform

import numpy as np
import cv2
cap = cv2.VideoCapture('3.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# fgbg = cv2.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2()

erodeKernel = np.ones((5,5),np.uint8)
dilateKernel = np.ones((7,1),np.uint8)

# more sizeof kernel is making less noise but after a reach, its reducing main foreground too
# element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

thetaPrev = 0

while(1):
    ret, frame = cap.read()
    skel = np.zeros(frame.shape,np.uint8)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)

    
    # Opening is just another name of erosion followed by dilation
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # It erodes away the boundaries of foreground object | A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
    erosion = cv2.erode(fgmask,erodeKernel,iterations=1)
    
    erosion = cv2.dilate(fgmask, kernel, iterations = 1)
    
    #for boundry derivative - no need coz using canny boundry detector
    # sobelx = cv2.Sobel(fgmask,cv2.CV_64F,1,0,ksize=3)
    # sobely = cv2.Sobel(fgmask,cv2.CV_64F,0,1,ksize=3)
    # laplacian = cv2.Laplacian(fgmask,cv2.CV_64F)

    # for canny edge detector | This makes the width of boundry = 1 | So erode after canny should not be used
    edges = cv2.Canny(fgmask,50,150,apertureSize = 3)
    
    # This will make the lines thicker which will help fit the Hough lines better | This is practically not working
    #edges = cv2.dilate(edges,dilateKernel,iterations = 1)
    

    lines = cv2.HoughLines(edges,1,np.pi/180,100) # last param is the threshold

    # if lines is not None:
    #     for line in lines:
    #         for rho,theta in lines[0]:
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))

    #             # now the line is generated on to the original colored frame
    #             cv2.line(frame,(x1,y1),(x2,y2),(255,40,10),2)
    

    rhoAvg = 0
    count = 1
    thetaAvg = 0
    if lines is not None:
        for line in lines:
            rhoAvg = line[0][0]
            thetaAvg = line[0][1]
            for rho,theta in line:
                rhoAvg = (rhoAvg * count + rho)/(count+1)
                thetaAvg = (thetaAvg * count + theta)/(count+1)
                count += 1

    # if deviation in theta is more than threshold take prev value
    if(abs((thetaPrev-thetaAvg)/(np.pi)*180)>20):
        a = np.cos(thetaPrev)
        b = np.sin(thetaPrev)
    else:
        a = np.cos(thetaAvg)
        b = np.sin(thetaAvg)
    
    x0 = a*rhoAvg
    y0 = b*rhoAvg
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    if rhoAvg != 0 and thetaAvg != 0:
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
    thetaPrev = thetaAvg

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

# How to retain edges in every frame,since all frames are different - Done by a,b,c,d
