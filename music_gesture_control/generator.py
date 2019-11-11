import matplotlib.pyplot as plt
import numpy
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2() 
count = 1

while True:
	# read the current frame
	ret, frame = cap.read()
	if not ret:
		print ("Unable to capture video")
		break 

	fgmask = fgbg.apply(frame) 
	# cv2.imshow('fgmask', frame)
	cv2.imshow('frame', fgmask)
	if count%30==1:
		# cv2.imshow("Frame", frame)
		cnt = (int)(count/30)
		print("./binImages/next." + (str)(cnt) + ".png")
		
		# cv2.imwrite("./binImages/next." + (str)(cnt) + ".png", frame)
	elif count>270:
		break
	# cv2.imshow("Frame", frame)
	count += 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()