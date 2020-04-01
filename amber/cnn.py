import cv2
import csv
import numpy as np
#import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline

import os
import random

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split


nrows = 150
ncols = 150
channels = 3
# all
#possible_colors = ['Black', 'Silver', 'Gray', 'White', 'Yellow', 'Blue',
#                   'Red', 'Purple', 'Green', 'Pink', 'Brown', 'Tan', 'Orange']
# large
possible_colors = ['Black', 'Silver', 'Gray', 'White', 'Yellow', 'Blue',
                   'Red', 'Purple', 'Green', 'Brown', 'Tan', 'Orange']
# small
#possible_colors = ['Black', 'Silver', 'Gray', 'White', 'Yellow', 'Blue', 'Red', 'Green', 'Tan']

# A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images, labels):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    print("read_and_process_image")
    X = []  # images
    y = []  # labels
    #y = labels

    for image in list_of_images:
        #print(image)
        #print(cv2.imread(image, cv2.IMREAD_COLOR))
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncols), interpolation=cv2.INTER_CUBIC))
        # get the labels
#        for i in range(len(possible_colors)):
#            if possible_colors[i] in image:
#                y.append(i)
    for color in labels:
        y.append(possible_colors.index(color))
        
    print("exiting function")
    return X, y

# train_imgs = os.listdir('labeler/car_ims')
# train_imgs = ['car_ims/{}.jpg'.format(f'{x:06}') for x in range(1, 3001)]
print("start")
train_imgs = []
train_labels = []
with open('labeler/targets_large.csv') as handle:
    reader = csv.DictReader(handle)

    for row in reader:
        train_imgs.append('./labeler/car_ims/{}'.format(row['img'].strip()))
        train_labels.append(row['color'])

print("loaded imgs and labels")
X, y = read_and_process_image(train_imgs, train_labels)

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

ntrain = len(X_train)
nval = len(X_val)



batch_size = 32
columns = 5

print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)

window = (3, 3)
pool = (2, 2)

model = models.Sequential()
model.add(layers.Conv2D(batch_size, window, activation='relu', input_shape=(nrows,ncols,channels)))
model.add(layers.MaxPooling2D(pool))
model.add(layers.Conv2D(batch_size*2, window, activation='relu'))
model.add(layers.MaxPooling2D(pool))
model.add(layers.Conv2D(batch_size*4, window, activation='relu'))
model.add(layers.MaxPooling2D(pool))
model.add(layers.Conv2D(batch_size*4, window, activation='relu'))
model.add(layers.MaxPooling2D(pool))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(len(possible_colors), activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print('compiled model')


# Create the augmentation configuration
# Helps prevent overfitting
train_datagen = ImageDataGenerator(rescale=1./255)#,
#                                   rotation_range=40,
#                                   width_shift_range=0.2,
#                                   height_shift_range=0.2,
#                                   shear_range=0.2,
#                                   zoom_range=0.2,
#                                   horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)

print('datagens working')

# Create image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

print('generators generated')

#history = model.fit_generator(train_generator,
#                              steps_per_epoch=ntrain // batch_size,
#                              epochs=5,
#                              validation_data=val_generator,
#                              validation_steps=nval // batch_size)

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=1)
test_loss, test_acc = model.evaluate(X_val, y_val)

print('model fitted')

model.save_weights('model_weights.h5')
model.save('model_keras.h5')

print('model saved')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

ephocs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

X_test, y_test = read_and_process_image(test_imgs[0:10])
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)


i = 0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    text_labels.append(pred)
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()
