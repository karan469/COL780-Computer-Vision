import numpy as np
import cv2
import os

def bound(Min, Max, v):
    if v<Min:
        v = Min
    if v>Max:
        v = Max
    return v

def OneVideoOutLine(file):
    cap = cv2.VideoCapture(file)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    # fgbg = cv2.createBackgroundSubtractorGMG()
    fgbg = cv2.createBackgroundSubtractorMOG2()

    erodeKernel = np.ones((5,5),np.uint8)
    dilateKernel = np.ones((7,1),np.uint8)

    # more sizeof kernel is making less noise but after a reach, its reducing main foreground too
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    thetaPrev = 0

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    file = file.replace('.mp4', '.avi')
    out = cv2.VideoWriter('OutputLine' + file, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
    while(True):
        ret, frame = cap.read()
        if ret == True:
            skel = np.zeros(frame.shape,np.uint8)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(frame)

            
            # Opening is just another name of erosion followed by dilation
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            
            # It erodes away the boundaries of foreground object | A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
            erosion = cv2.erode(fgmask,erodeKernel,iterations=2)
            
            #erosion = cv2.dilate(fgmask, kernel, iterations = 1)
            
            #for boundry derivative - no need coz using canny boundry detector
            # sobelx = cv2.Sobel(fgmask,cv2.CV_64F,1,0,ksize=3)
            # sobely = cv2.Sobel(fgmask,cv2.CV_64F,0,1,ksize=3)
            # laplacian = cv2.Laplacian(fgmask,cv2.CV_64F)

            # for canny edge detector | This makes the width of boundry = 1 | So erode after canny should not be used
            edges = cv2.Canny(fgmask,50,150,apertureSize = 3)
            
            # This will make the lines thicker which will help fit the Hough lines better | This is practically not working
            #edges = cv2.dilate(edges,dilateKernel,iterations = 1)

            lines = cv2.HoughLines(edges,1,np.pi/180,100) # last param is the threshold
            #linesP = cv2.HoughLines(edges, 1, np.pi/180, 100, 5, 10)

            # dst = cv2.cornerHarris(fgmask, 2, 3, 0.04)
            # print(dst[0][0])
            if lines is not None:
                for line in lines:
                    for rho,theta in lines[0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))

                        # now the line is generated on to the original colored frame
                        cv2.line(frame,(x1,y1),(x2,y2),(255,40,10),2)

            # frame_flip = cv2.flip(frame,0)
            out.write(frame)
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def OneVideoOutLineSeg(file):
    cap = cv2.VideoCapture(file)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    kernel = np.ones((5,5),np.uint8)


    xmaxGlob = 0
    ymaxGlob = 0
    count = 0

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    file = file.replace('.mp4', '.avi')
    out = cv2.VideoWriter('Output'+file, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
    
    while(True):
        ret, frame = cap.read()
        if ret == True:
            skel = np.zeros(frame.shape,np.uint8)

            fgmask = fgbg.apply(frame)
            
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            erosion = cv2.erode(fgmask,kernel,iterations=2)
            fgmask = cv2.medianBlur(fgmask, 3)
            edges = cv2.Canny(fgmask,50,150,apertureSize = 3)
            lines = cv2.HoughLinesP(edges,1,np.pi/180,110,100,5)
            hlines = cv2.HoughLines(edges,1,np.pi/180,100)


            # try xmax and ymax from probabalistic - 1
            xmax = 0
            ymax = 0
            xmin = 10000
            ymin = 10000
            if lines is not None:
                for line in lines:
                    count+=1
                    for x1,y1,x2,y2 in line:
                        if max(x1,x2)>xmax:
                            xmax = max(x1,x2)
                        if max(y1,y2)>ymax:
                            ymax = max(y1,y2)
                        if min(x1,x2)<xmin:
                            xmin = min(x1,x2)
                        if min(y1,y2)<ymin:
                            ymin = min(y1,y2)
                    if count==1:
                        xmaxGlob = xmax
                        ymaxGlob = ymax    

            #--


            rhoAvg = 0
            count = 1
            thetaAvg = 0
            if hlines is not None:
                for hline in hlines:
                    rhoAvg = hline[0][0]
                    thetaAvg = hline[0][1]
                    for rho,theta in hline:
                        rhoAvg = (rhoAvg * count + rho)/(count+1)
                        thetaAvg = (thetaAvg * count + theta)/(count+1)
                        count += 1

            a = np.cos(thetaAvg)
            b = np.sin(thetaAvg)
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
            
            cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            # cv2.line(frame, (min(x1,xmax),min(y1,ymax)), (min(x2,xmax),min(y2,ymax)), (255,0,0), 2)

            # if hlines is not None:
            #     for hline in hlines:
            #         for rho,theta in hline:
            #             a = np.cos(theta)
            #             b = np.sin(theta)
            #             x0 = a*rho
            #             y0 = b*rho
            #             x1 = int(x0 + 1000*(-b))
            #             y1 = int(y0 + 1000*(a))
            #             x2 = int(x0 - 1000*(-b))
            #             y2 = int(y0 - 1000*(a))
                        
            #             x1 = bound(xmin, xmax, x1)
            #             x2 = bound(xmin, xmax, x2)
            #             y1 = bound(ymin, ymax, y1)
            #             y2 = bound(ymin, ymax, y2)
            #             cv2.line(frame, (min(x1,xmax),min(y1,ymax)), (min(x2,xmax),min(y2,ymax)), (255,0,0), 2)

            xmaxGlob = xmax
            ymaxGlob = ymax
            out.write(frame)
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def outAllLineVersion():
    for file in os.listdir("/Users/Karan/Desktop/visiona1/"):
        print(file)
        if file.endswith(".mp4"):
            path=os.path.join("/Users/Karan/Desktop/visiona1/", file)
            OneVideoOutLine(file)

def outAllLineSegVersion():
    for file in os.listdir("/Users/Karan/Desktop/visiona1/"):
        print(file)
        if file.endswith(".mp4"):
            path=os.path.join("/Users/Karan/Desktop/visiona1/", file)
            OneVideoOutLineSeg(file)

# OneVideoOutLineSeg('2.mp4')
