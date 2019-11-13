import matplotlib.pyplot as plt
import numpy as np
import cv2

cap1 = cv2.VideoCapture('./data/part2_videos/train_next1.webm')
cap2 = cv2.VideoCapture('./data/part2_videos/train_next2.webm')
# cap3 = cv2.VideoCapture('./data/part2_videos/next3.webm')
# cap4 = cv2.VideoCapture('./data/part2_videos/next4.webm')

# cap2 = cv2.VideoCapture('next2.webm')
# cap3 = cv2.VideoCapture('next3.webm')
# cap4 = cv2.VideoCapture('next4.webm')

fgbg = cv2.createBackgroundSubtractorMOG2() 
count = 1
index = 1
# kernel = np.ones((3,3),np.uint8)/9
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

while True:
	# read the current frame
	ret, frame = cap1.read()
	if not ret:
		print ("Unable to capture video")
		break 

	if count%5!=0:
		count +=1
		continue

	fgmask = fgbg.apply(frame)
	f1 = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	f2 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	new = cv2.resize(f2, (50, 50))
	edges = cv2.Canny(new,100,200)
    
	cv2.imwrite("./data/test2_final/next." + (str)(index) + ".png", edges)

	# cv2.imwrite("./data/train1/next." + (str)(count) + ".png", frame)

	print("./data/test2_final/next." + (str)(index) + ".png")
	count += 1
	index +=1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap1.release()

while True:
	# read the current frame
	ret, frame = cap2.read()
	if not ret:
		print ("Unable to capture video")
		break 

	if count%5!=0:
		count +=1
		continue

	fgmask = fgbg.apply(frame)
	f1 = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
	f2 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	new = cv2.resize(f2, (50, 50))
	edges = cv2.Canny(new,100,200)
    
	cv2.imwrite("./data/test2_final/next." + (str)(index) + ".png", edges)

	# cv2.imwrite("./data/train1/next." + (str)(count) + ".png", frame)

	print("./data/test2_final/next." + (str)(index) + ".png")
	count += 1
	index +=1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap2.release()


# while True:
# 	# read the current frame
# 	ret, frame = cap3.read()
# 	if not ret:
# 		print ("Unable to capture video")
# 		break 

# 	if count%5!=0:
# 		count +=1
# 		continue

# 	fgmask = fgbg.apply(frame)
# 	f1 = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
# 	f2 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

# 	new = cv2.resize(f2, (50, 50))
# 	edges = cv2.Canny(new,100,200)
    
# 	cv2.imwrite("./data/test2_final/next." + (str)(index) + ".png", edges)


# 	# cv2.imwrite("./data/train1/others." + (str)(count) + ".png", frame)

# 	print("./data/train2_final/others." + (str)(index) + ".png")
# 	count += 1
# 	index +=1
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cap3.release()


# while True:
# 	# read the current frame
# 	ret, frame = cap4.read()
# 	if not ret:
# 		print ("Unable to capture video")
# 		break 

# 	if count%5!=0:
# 		count +=1
# 		continue

# 	fgmask = fgbg.apply(frame)
# 	f1 = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
# 	f2 = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

# 	new = cv2.resize(f2, (50, 50))
# 	edges = cv2.Canny(new,100,200)
    
# 	cv2.imwrite("./data/train2_final/others." + (str)(index) + ".png", edges)


# 	# cv2.imwrite("./data/train1/others." + (str)(count) + ".png", frame)

# 	print("./data/train2_final/others." + (str)(index) + ".png")
# 	count += 1
# 	index +=1
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cap4.release()


# while True:
# 	# read the current frame
# 	ret, frame = cap2.read()
# 	if not ret:
# 		print ("Unable to capture video")
# 		break 

# 	cv2.imwrite("./data/train1/others." + (str)(count) + ".png", frame)
# 	print("./data/train1/others." + (str)(count) + ".png")

# 	count += 1
	
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cap2.release()

# while True:
# 	# read the current frame
# 	ret, frame = cap3.read()
# 	if not ret:
# 		print ("Unable to capture video")
# 		break 

# 	cv2.imwrite("./data/train1/others." + (str)(count) + ".png", frame)
# 	print("./data/train1/pause." + (str)(count) + ".png")

# 	count += 1
	
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cap3.release()

# while True:
# 	# read the current frame
# 	ret, frame = cap4.read()
# 	if not ret:
# 		print ("Unable to capture video")
# 		break 

# 	cv2.imwrite("./data/train1/prev." + (str)(count) + ".png", frame)
# 	print("./data/train1/prev." + (str)(count) + ".png")

# 	count += 1
	
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cap4.release()

cv2.destroyAllWindows()


