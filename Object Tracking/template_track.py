# import keras
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt

template = cv2.imread('./template.jpg',0)
w, h = template.shape[::-1]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img_rgb = frame
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    cv2.imshow('frame', img_rgb)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print('Hello world')
