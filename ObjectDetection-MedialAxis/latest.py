import numpy as np
import cv2
cap = cv2.VideoCapture('1.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

kernel = np.ones((5,5),np.uint8)


xmaxGlob = 0
ymaxGlob = 0
count = 0
while(1):
    ret, frame = cap.read()
    skel = np.zeros(frame.shape,np.uint8)

    fgmask = fgbg.apply(frame)
    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    erosion = cv2.erode(fgmask,kernel,iterations=1)
    edges = cv2.Canny(fgmask,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,110,100,5)
    hlines = cv2.HoughLines(edges,1,np.pi/180,100)

    # if lines is not None:
    #     for line in lines:
    #         for rho,theta in line:
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))

    #             cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
    #cv2.drawContours(erosion, contours, -1, (0,255,0), 3)

    # try xmax and ymax from probabalistic - 1
    xmax = 0
    ymax = 0
    if lines is not None:
        for line in lines:
            count+=1
            for x1,y1,x2,y2 in line:
                if max(x1,x2)>xmax:
                    xmax = max(x1,x2)
                if max(y1,y2)>ymax:
                    ymax = max(y1,y2)
            if count==1:
                xmaxGlob = xmax
                ymaxGlob = ymax    

    # if abs(xmax-xmaxGlob)>5:
    #     xmax = int((xmaxGlob+3*xmax)/4)
    # if abs(ymax-ymaxGlob)>5:
    #     ymax = int((ymax+3*xmax)/4)

    #print(fgmask)


    #----------------------------------- finding last row where no white pixel
    # flagZeroRow = 0
    # for r in range(len(fgmask)):
    #     flagZeroRow = 0
    #     for c in range(len(fgmask[0])):
    #         # if fgmask[r][c]==0:
    #         #     print('true')
    #         #     print(fgmask[r][c])
    #         if fgmask[r][c]!=0:
    #             flagZeroRow = 1
    #             break
    #     if flagZeroRow==0:
    #         ymax = r
    #         break

    # for c in range(len(fgmask[0])):
    #     flagZeroRow = 0
    #     for r in range(len(fgmask)):
    #         if fgmask[r][c]!=0:
    #             flagZeroRow = 1
    #             break
    #     if flagZeroRow==0:
    #         xmax = c
    #         break
    #----------------------------


    #--
    if hlines is not None:
        for line in hlines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(frame, (min(x1,xmax),min(y1,ymax)), (min(x2,xmax),min(y2,ymax)), (255,0,0), 2)

    # if lines is not None:
    #     for line in lines:
    #         print(line)
    #         for x1,y1,x2,y2 in line:
    #             cv2.line(frame, (min(x1,xmax),min(y1,ymax)), (min(x2,xmax),min(y2,ymax)), (255,0,0), 2)

    xmaxGlob = xmax
    ymaxGlob = ymax
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
