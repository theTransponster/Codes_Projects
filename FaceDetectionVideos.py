# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

! git clone https://github.com/opencv/opencv.git

#video='/content/drive/My Drive/Protocol_4/2/Train/1_1_01_1.avi'
video='/content/7C2DA86B-0CFA-4148-98BF-68FF77EA2FD0.mov'

import cv2

import numpy as np
from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture(video)

if (cap.isOpened()== False):

  print("Error opening video stream or file")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

cap.set(1,50); # Where frame_no is the frame you want
ret, frame = cap.read() # Read the frame
#cv2_imshow(frame) # show frame on window
face_cascade = cv2.CascadeClassifier('/content/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#cap.release()
#cv2.destroyAllWindows()

"""*Image visualization example*"""

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
cv2_imshow(frame)

imgCrop = frame[y:y+h,x:x+w]
cv2_imshow(imgCrop)

"""**FOLDER GENERATION AND LABELING**"""

import glob
import os
from os.path import basename
labels=[]

def read_vid(vid_list, videos):
    #print (os.path.split(videos)[1])
    name=os.path.split(videos)[1].split('_')
    identificador=name[3]
    if identificador=='1.avi':
      labels.append(1) #1 is for real face
      print(identificador)
    else:
      labels.append(0) #0 is for fake
    n = cv2.VideoCapture(videos)
    vid_list.append(n)
    return vid_list

path = glob.glob("/content/drive/My Drive/Protocol_4/1/Train/*.avi")
list_v = []

cv_image = [read_vid(list_v, videos) for videos in path]

import random
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

lista = range(0,length-1)
al=random.sample(lista,k=10)

"""**FRAMES PREPROCESSING**"""

list_videos=[]
for j in len(list_v):
  cap = cv2.VideoCapture(list_v[j])
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  lista = range(0,length-1)
  al=random.sample(lista,k=10)
  list_frames = []
  for i in len(al):
    cap.set(1,i); # Where frame_no is the frame you want
    ret, frame = cap.read() # Read the frame
    #cv2_imshow(frame) # show frame on window
    face_cascade = cv2.CascadeClassifier('/content/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    #cap.release()
    #cv2.destroyAllWindows()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    imgCrop = frame[y:y+h,x:x+w]
    list_frames.append(imgCrop)
    #cv2_imshow(frame)
  list_videos.append(list_frames)

# Guardar las imagenes

for i in range(len(list_frames)):
  cv2.imwrite(i+'.png',list_frames[i])

"""**NEURAL NETWORK**

*Resize images*
"""

import cv2
def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 224, 224, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

x_train_resized = resize_data(train_set_x_orig)
x_test_resized = resize_data(test_set_x_orig)

print('x_train shape:', x_train_resized.shape)
print(x_train_resized.shape[0], 'train samples')
print(x_test_resized.shape[0], 'test samples')

train_set_x = x_train_resized/255.
test_set_x = x_test_resized/255.

train_set_y=train_set_y.T
test_set_y=test_set_y.T

"""*Keras model*"""

import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)

