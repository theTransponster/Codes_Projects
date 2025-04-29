# -*- coding: utf-8 -*-


import keras
keras.__version__

import os, shutil

# Please correct the urls accordingly

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = 'train'

# The directory where we will
# store our smaller dataset
base_dir = 'data3'
os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

x_batch, y_batch = next(train_generator)
print(y_batch)
x_val, y_val=next(validation_generator)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""#  Part 2: Adjusting hyperparameters and regularization method

First we need to understand and to analyse the results obtained with the above first trial, in order to be able to correct the hyperparameters and to add the most suitable regularization method.

For the hyperparameters we have:

* Optimizer method: SGD; ussually works well for large datasets and is efficient for classification problems.
* Batch size: 20; a small batch can lead to an inaccurate gradient estimation since it will provoke more fluctuation.
* Regularization: no method is implemented, which can be a problem related with overfitting.
* Learning rate: 0.01;
    * this hyperparameter can be tricky, since is a relatively large value, and although this will require less training epoch, it can converge too quickly leading to a suboptimal result, and a loss value oscillating over epochs.
* Hidden layers: 4;
    * can be consider as an addecuate value to avoid overfitting, although the addition of more layers can be beneficial for the training set.

Then, the results analysis:

* Loss: 0.4496
* Val_loss: 1.1556
    * The val_loss (corresponing to the test set), is bigger than the loss value (for the trainig set), which is
      opposite to what is desirable, because, if the loss is small (like in this case), it could indicate a good
      behavior of the model, but also, it can be a sign of overfitting, as is happening here, where the error for
      the test set is large. This can be due to the learning rate selected (the model converge quick), or because
      of the abscence of a regularizer method.
* Accuracy: 0.7960
* Val_acc: 0.5910
    * The accuracy value is not that bad, however, is bigger for the training set than for the validation set,
      which again, can be a sign of overfitting.

* With the plots above, we can see that the model has a behavior with a high variance and bias.

For this part, the datasets are already charged, then we can proceed to change the
* Number of hidden layers
* Number of hidden units
* Learning rate
* Mini-batch size

in the following code parts

# ***Test1***

* Hidden layers: 6
* Hidden units: from 512 to 256

*Here, we add more layers to try to increase the accuracy of model, decreasing at the same time the number of hidden units to avoid overfitting.*
"""

model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

"""Then, we change the learning rate from 0.01 to 0.005


*We decrease the leraning rate to moderate number (half size)*
"""

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['acc'])

"""And the batch size from 20 to 50:

*We are incrementing the batch size to avoid possible fluctuation in the gradient estimation.*
"""

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=50,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=50,
        class_mode='binary')

"""Finally, the epochs are run and the results are plotted:"""

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""**Results discussion**

With this test we can notice that:

Pros:
* The loss decreased: from 0.4496 to 0.2862
* The val_loss also decreased: from 1.1554 to 0.7818
* The accuracy increased: from 0.7960 to 0.8900

Cons:
* The val_acc decreased: from 0.5910 to 0.5752
* The val_loss is still bogger than the loss, which indicates still an overfitt in the model.
* The plots still indicates a high variance and bias (but a little less than in the first trial).

Up to here, we can understand that the hyperparameters change was useful at some point, mostly because of the accuracy increase, however, it could be useful to add the regularization method to improve more the model.

Also, an increment on the epoch numbers could be useful (since we decrease the learning rate).

# ***Test 2- Dropout regularization***

In this case, a regularization method will be also added, in addition to above hyperparameters modification.

Based on the Keras documentation (https://keras.io/regularizers/)

Again, the hyperparameters are:

* Hidden layers: 6
* Hidden units: from 512 to 256
* Learning rate from 0.01 to 0.005
* Batch size from 20 to 50

And the regularization method is based on the "DROPOUT" on every layer (which can be useful to prevent the model overfitting)
"""

# From keras:

from keras import regularizers
from keras import optimizers

#The arguments for the dropout are:
#keras.layers.Dropout(rate, noise_shape=None, seed=None)
#Where:
#rate: float between 0 and 1. Fraction of the input units to drop.
#noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will
#be multiplied with the input. For instance, if your inputs have shape  (batch_size, timesteps, features)
#and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
#seed: A Python integer to use as random seed.


# 1- Layers setup
model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.8,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.8,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.8,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.8,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.8,noise_shape=None, seed=None))
model.add(layers.Dense(1, activation='sigmoid'))

#2- Learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['acc'])

#3- Batch size
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=50,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=50,
        class_mode='binary')


#4- Epoch

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

# 5- Results plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""**Results discussion**

This is a clare example of an overfitting model.

* The accuracy valu decreased even from the first trial (from 0.7960 to 0.5028).
* The variance and bias increased considerably for the training and validation part.

The main reason of this result can be due to the probability setted for the dropout model (0.8):
(https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275)


* The regularization parameter, p(1-p) is maximum at p = 0.5.
* The dropout rate argument is (1-p). For intermediate layers, choosing (1-p) = 0.5 for large networks is ideal.
* For the input layer, (1-p) should be kept about 0.2 or lower. This is because dropping the input data can adversely affect the training.
* A (1-p) > 0.5 is not advised, as it culls more connections without boosting the regularization.

# ***Test 3- Dropout lower values***


This case will be the last trial with the hyperparameters:
* Hidden layers: 6
* Hidden units: from 512 to 256
* Learning rate from 0.01 to 0.005
* Batch size from 20 to 50

And with the dropout regularization method, but this time, the value will be lower: 0.5 (based on the above explanation).
"""

# 1- Layers setup
model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(1, activation='sigmoid'))

#2- Learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['acc'])

#3- Batch size
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=50,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=50,
        class_mode='binary')


#4- Epoch

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

# 5- Results plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""**Results discussion**

The results are much better than the ones obtained in the Test2.

Pros:
* The val_loss (0.6838) is smaller than the loss (0.6592): can indicate that the model is not overfitting, since is showing best results (less error) for the test set.
* The accuracy (0.6168) is better than in test 2 (0.5028).

Cons:
* The loss value (0.6592) is higher than in first trial (0.4496).
* The accuracy (0.6168) is smaller than in first trial (0.7960).

From the plots:
* The variance is still high, but the bias is much slower, both for the accuracy and for the loss.

# ***Test 4- Regularization- Kernel L2 penalty***

This time, we will use a different regularization method (https://keras.io/regularizers/).

Also the hyperparameters will be changed:

* Hidden layers: 5
* Hidden units: will vary
* Learning rate: 0.005 but the number of epoch will be increased (from 30 to 40).
* Batch size: 30
* l2 penalty: 0.01
"""

# 1- Layers setup
model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1, activation='sigmoid'))

#2- Learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['acc'])

#3- Batch size
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=30,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=30,
        class_mode='binary')


#4- Epoch

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=40,
      validation_data=validation_generator,
      validation_steps=50)

# 5- Results plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""**Results discussion**

We have problems again, related with overfitting.

Pros:
* The accuracy increased considerably: 0.9275

Cons:
* The loss and val_loss are much bigger than i past tests: 0.9939 and 1.7848

From the plot:
* The variance is high, indicating overfitting.

# ***Test 5- Regularization- Kernel l2 penalty, bias and activation***

This time, we will return to some values from first test;

For hyperparameters:
* Hidden units: will vary
* Epoch number: 30

And will change:
* Learning rate: 0.002
* Hidden layers: 7
* l2 penalization: 0.05
* Validation steps: 60

As well as the addition of bias and activity regularizers (https://keras.io/regularizers/ ), in different layers.
"""

# 1- Layers setup
model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
model.add(layers.Dense(256, bias_regularizer=regularizers.l1(0.01)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.05)))
model.add(layers.Dense(1, activation='sigmoid'))

#2- Learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.002, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['acc'])

#3- Batch size
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=30,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=30,
        class_mode='binary')


#4- Epoch

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=60)

# 5- Results plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""**Results analysis**

* Is the test with more bias and variance for the training and validation accuracy.
* Even though the val_acc (0.4955) is bigger than the accuracy (0.4880), both values are too small.
* The loss and val_loss are too high: 4.7348 and 4.6825

# Test 6

As the best behavior was obtained with the model of "Test 3" (with a dropout regularization method), and the best accuracy values was resulted from "Test 4", this trial will be based on the combination of both.

* Hidden layers: 5
* Hidden units: will vary
* Batch size: 50
* Dropout method: p=0.5
* Eppoch number: 30
* Learning rate: 0.005
"""

# 1- Layers setup
model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5,noise_shape=None, seed=None))
model.add(layers.Dense(1, activation='sigmoid'))

#2- Learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['acc'])

#3- Batch size
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=50,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=50,
        class_mode='binary')


#4- Epoch

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

# 5- Results plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

