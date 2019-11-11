import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture('next1.webm')
fgbg = cv2.createBackgroundSubtractorMOG2() 
count = 1
kernel = np.ones((3,3),np.uint8)/9
while True:
	# read the current frame
	ret, frame = cap.read()
	if not ret:
		print ("Unable to capture video")
		break 

	# fgmask = fgbg.apply(frame) 
	# fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	# fgmask = cv2.medianBlur(fgmask, 3)
	# cv2.imshow('fgmask', frame)
	
	# cv2.imshow('frame', fgmask)

	cv2.imwrite("./binImages/next." + (str)(cnt) + ".png", frame)
	print("./binImages/next." + (str)(cnt) + ".png")

	# if count%30==1:
	# 	# cv2.imshow("Frame", frame)
	# 	cnt = (int)(count/30)
	# 	print("./binImages/next." + (str)(cnt) + ".png")
		
	# 	cv2.imwrite("./binImages/next." + (str)(cnt) + ".png", frame)
	
	# elif count>270:
	# 	break
	
	count += 1
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
