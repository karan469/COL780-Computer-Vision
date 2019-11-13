#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    print('Supressed Warnings..')
from keras.models import load_model
from keras.models import model_from_json
import json
from keras import optimizers


# In[2]:


with open('model_in_json_back.json','r') as f:
    model_json = json.load(f)


# In[3]:


model = model_from_json(model_json)
model.load_weights('model_weights_back.h5')


# In[4]:


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[5]:


import cv2
import numpy as np
import os
cap = cv2.VideoCapture(0)

count = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,400)
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2
kernel = np.ones((1,1),np.uint8)
trash_dest = './bin/'
while(True):
    ret, frame = cap.read()
    if not ret:
        print ("Unable to capture video")
        break
 
    new = cv2.resize(frame, (50, 50))
    edges = cv2.Canny(new, 100, 200)
#     cv2.imshow('Edges', edges)
    cv2.imwrite(trash_dest + str(count) + '.png', edges)
    edges = cv2.imread(trash_dest + str(count) + '.png')
    os.system('rm ' + trash_dest + str(count) + '.png')
#     print(edges.shape)
    edges = np.reshape(edges,[1,50,50,-1])
#     edges = np.append(edges, edges)
    
    classes = model.predict_classes(edges)
#     classes = []
#     classes.append([2])
    
    text = ''
    print(classes)
    if classes[0] == [0]:
        text = 'Next'
    elif classes[0] == [1]:
        text = 'None'
    elif classes[0] == [2]:
        text = 'Pause'
    elif classes[0] == [3]:
        text = 'Prev'
    
    cv2.putText(frame,text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imshow('Prediction', frame)
    
    count += 1
#     if count > 500:
#         break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




