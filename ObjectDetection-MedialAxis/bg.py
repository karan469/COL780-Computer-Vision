import numpy as np
import cv2
import os
import math

def bound(Min, Max, v):
    if v<Min:
        v = Min
    if v>Max:
        v = Max
    return v

cap = cv2.VideoCapture('5.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# fgbg = cv2.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((3,3),np.uint8)/9

rhoPrev = 0
thetaPrev = 0
while(True):
        ret, frame = cap.read()
        if ret == True:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.medianBlur(fgmask, 3)
            erosion = cv2.erode(fgmask,kernel,iterations=2)

            blur = cv2.GaussianBlur(erosion,(3,3),0)
            edges = cv2.Canny(blur,50,150,apertureSize = 3)


            linesP = cv2.HoughLinesP(edges,1,np.pi/180, threshold = 100, minLineLength = 100, maxLineGap = 35)

            xmax = 0
            ymax = 0
            xmin = 10000
            ymin = 10000
            if linesP is not None:
                for line in linesP:
                    for x1,y1,x2,y2 in line:
                        #cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        if max(x1,x2)>xmax:
                            xmax = max(x1,x2)
                        if max(y1,y2)>ymax:
                            ymax = max(y1,y2)
                        if min(x1,x2)<xmin:
                            xmin = min(x1,x2)
                        if min(y1,y2)<ymin:
                            ymin = min(y1,y2)

            itr = 0
            rhoAvg = 0
            thetaAvg = 0
            if linesP is not None:
                for lineP in linesP:
                    count = 0
                    for x1,y1,x2,y2 in lineP:
                        if x2!=x1:
                            theta = math.atan((y2-y1)/(x2-x1))
                        else:
                            theta = 90*(np.pi)/180
                        print(theta)
                        rho = (abs(x1*math.tan(theta)-y1))/math.sqrt(1+math.tan(theta)*math.tan(theta))

                        thetaAvg = (theta + thetaAvg*count)/(count+1)
                        rhoAvg = (rho + rhoAvg*count)/(count+1)
                        # cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                    count += 1
            if itr!=0:
                if(abs(rhoPrev - rhoAvg) > 2):
                    rhoAvg = (rhoPrev + rhoAvg)/2;
                if(abs(thetaPrev - thetaAvg) > 5):
                    thetaAvg = (thetaPrev + thetaAvg)/2;
            rhoPrev = rhoAvg
            thetaPrev = thetaAvg

            itr +=1
            print(thetaAvg)
            
            a = np.cos(-(np.pi)/2 + thetaAvg)
            b = np.sin(-(np.pi)/2 + thetaAvg)
            x0 = a*rhoAvg
            y0 = b*rhoAvg
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            x1 = bound(xmin, xmax, x1)
            x2 = bound(xmin, xmax, x2)
            y1 = bound(ymin, ymax, y1)
            y2 = bound(ymin, ymax, y2)
            
            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()