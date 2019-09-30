from PIL import Image
import os,sys
import glob


user_input = raw_input("Enter the path of your file: ")
assert os.path.exists(user_input), "I did not find the file at, "+str(user_input)
inn = user_input + '*.jpg'
inn = glob.glob(inn)

for i in range(0,len(inn)):
	image = Image.open(inn[i])
	newImage = image.resize(image.size)
	newImage.save(user_input + inn[i].split(user_input)[1].split('.jpg')[0] + 'new' + '.jpg')
	print(image.size)
print(inn)