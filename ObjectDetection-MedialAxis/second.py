#without morphological transformation

# import numpy as np
# import cv2
# cap = cv2.VideoCapture('1.mp4')
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# # fgbg = cv2.createBackgroundSubtractorGMG()
# fgbg = cv2.createBackgroundSubtractorMOG2()
# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()

#--------------------------------#

# with morphological transform

import numpy as np
import cv2
cap = cv2.VideoCapture('1.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((5	,5),np.uint8)
# more sizeof kernel is making less noise but after a reach, its reducing main foreground too
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    # Opening is just another name of erosion followed by dilation
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # it erodes away the boundaries of foreground object | A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
    fgmask = cv2.erode(fgmask,kernel,iterations=1)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()