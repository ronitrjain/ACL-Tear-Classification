import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('kneemridataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Preprocessing
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import random

img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
batch_size = 32

# Define the hyperparameters
learning_rate = 0.001
epochs = 50
steps_per_epoch = 100

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

# Loading data from directories
train_generator = train_datagen.flow_from_directory(
        'kneemridataset/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'kneemridataset/test',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

# Building the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fitting the model
history = model.fit(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_data=test_generator,
      validation_steps=50)

# Evaluating the model
_, acc = model.evaluate(test_generator, steps=len(test_generator), verbose=0)
print('> %.3f' % (acc * 100.0))

# Saving the model
model.save('acl_classifier.h5')
