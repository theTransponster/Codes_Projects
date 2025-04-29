# -*- coding: utf-8 -*-


# 1 - Implementing the CNN architecture
"""

import keras
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import normalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D




#Model considering a multiclass single labeled problem (considering non-cat as the second class); activation Relu
#Normalization layers commented


model=Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='valid', activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
#model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
#model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

"""# 2 - Testing with cat vs. non cat dataset

**Image preprocessing**
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

import cv2
def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 224, 224, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

x_train_resized = resize_data(train_set_x_orig)
x_test_resized = resize_data(test_set_x_orig)
#x_train_resized = x_train_resized / 255
#x_test_resized = x_test_resized / 255

print('x_train shape:', x_train_resized.shape)
print(x_train_resized.shape[0], 'train samples')
print(x_test_resized.shape[0], 'test samples')

train_set_x = x_train_resized/255.
test_set_x = x_test_resized/255.

train_set_y=train_set_y.T
test_set_y=test_set_y.T

"""**Training model**"""

model.fit(train_set_x, train_set_y,
    batch_size=20,
    epochs=3,
    validation_data=(test_set_x, test_set_y))


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

model.fit(train_set_x, train_set_y,
    steps_per_epoch=5,
    epochs=3,
    validation_data=(test_set_x, test_set_y),
    validation_steps=5)


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

"""As we can observe, the model is not behaving correctly, maybe because we are using the relu and softmax activation for a binary classification, so let's try with the sigmod function.

**Sigmoid activation in last layer**
"""

#Testing with sigmoid as the output layer since is a binary classification
#Also using the normalization layers.


model=Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='valid', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='sigmoid'))
model.add(layers.Dense(4096, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(train_set_x, train_set_y,
    batch_size=20,
    epochs=10,
    validation_data=(test_set_x, test_set_y))


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

"""The big accuracy on the training, instead on the validation, could indicate that the model is overfitted.

The model behaves a little bit better with the sigmoid activation, however, a big variance is still observed, so let's try with some optimizers and regularization layers.
"""

# 1- Adding dropout 0.5
# 2- Change optimizer to sgd

model=Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='valid', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#number of epochs is little in order to have time to run multiple tests.

model.fit(train_set_x, train_set_y,
    steps_per_epoch=5,
    epochs=3,
    validation_data=(test_set_x, test_set_y),
    validation_steps=5)


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

# 1- Adding L2 optimizer
# 2- Change optimizer to Adagrad
# 3- No specific activation on the last layers

from keras import regularizers

model=Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='valid', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(4096))
model.add(layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adagrad(learning_rate=0.01)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(train_set_x, train_set_y,
    steps_per_epoch=5,
    epochs=3,
    validation_data=(test_set_x, test_set_y),
    validation_steps=5)


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

"""This type of model increased a lot the loss value, so let's use only the dropout regularizer method"""

# 1- Change optimizer to Adam


from keras import regularizers

model=Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='valid', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='sigmoid'))
model.add(layers.Dense(4096, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(train_set_x, train_set_y,
    steps_per_epoch=5,
    epochs=3,
    validation_data=(test_set_x, test_set_y),
    validation_steps=5)


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

# Test with SGD and dropout at 0.2

model=Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='valid', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(train_set_x, train_set_y,
    steps_per_epoch=5,
    epochs=3,
    validation_data=(test_set_x, test_set_y),
    validation_steps=5)


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

# Test with SGD and dropout at 0.2, but checking behaviour with a softmax activation

model=Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='valid', activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=3,strides=2))
model.add(Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(train_set_x, train_set_y,
    steps_per_epoch=5,
    epochs=3,
    validation_data=(test_set_x, test_set_y),
    validation_steps=5)


# Score trained model.
scores = model.evaluate(test_set_x, test_set_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']


epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()



plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

