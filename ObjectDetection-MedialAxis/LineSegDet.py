import cv2
import numpy as np

cap = cv2.VideoCapture('3.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erodeKernel = np.ones((5,5),np.uint8)

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # It erodes away the boundaries of foreground object | A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
    erosion = cv2.erode(fgmask,erodeKernel,iterations=2)
    lsd = cv2.createLineSegmentDetector(0)
    #lsd = cv2.createLineSegmentDetector.detect(fgmask, lines, width, prec, nfa)
    lines = lsd.detect(fgmask)[0] 
    drawn_img = lsd.drawSegments(frame,lines)
    cv2.imshow("LSD",drawn_img )
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



# img = cv2.imread("test.png",0)

# #Create default parametrization LSD

# #Detect lines in the image
# lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines

# #Draw detected lines in the image
# drawn_img = lsd.drawSegments(img,lines)

# #Show image
# cv2.imshow("LSD",drawn_img )
# cv2.waitKey(0)