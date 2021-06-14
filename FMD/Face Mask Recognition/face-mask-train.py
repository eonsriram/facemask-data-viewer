import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

import time



wm = os.listdir("Dataset/without_mask")
m = os.listdir("Dataset/with_mask")


# Pre-Processing the image
def ppimg(path):
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img1 = np.array(img)
    img2 = np.array(img).reshape(1, 100, 100, 3)
    return img1, img2


wmdata = []
mdata = []

for img in wm:
    img = cv2.imread('/content/drive/MyDrive/colabutils/Face Mask Detection/{}/without_mask/'.format(data) + img,
                     cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img = np.array(img)
    wmdata.append(img)

wmdata = np.array(wmdata)

for img in m:
    img = cv2.imread('/content/drive/MyDrive/colabutils/Face Mask Detection/{}/with_mask/'.format(data) + img,
                     cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img = np.array(img)
    mdata.append(img)

mdata = np.array(mdata)

dataset = np.concatenate((wmdata, mdata))
y = np.array([0] * len(wmdata) + [1] * len(mdata))

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.1, random_state=42)
y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))

wmdata = wmdata.astype('float32')
mdata = mdata.astype('float32')

wmdata /= 255
mdata /= 255

shape = (100, 100, 3)

model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(10, kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(x=X_train, y=y_train, epochs=5)

#model.save('Saved Model')

model.evaluate(X_test, y_test)

#model.predict(ppimg(img)[1])[0][0]
