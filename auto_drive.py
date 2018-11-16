import csv
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import random
import scipy.ndimage


random.seed(435)


print("reading data")
if os.path.exists('np_data_imgages.npy'):
    try:
        X_train = np.load('np_data_imgages.npy')
    except:
        print("'np_data_imgages' file not found. please run submission.IPYNB file, to generate data")
        exit()


if os.path.exists('np_data_maesurments.npy'):
    try:
        y_train = np.load('np_data_maesurments.npy')
    except:
        print("'np_data_maesurments' file not found. please run submission.IPYNB file, to generate data")
        exit()


print("starting")
model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(66, 200, 3)))
#model.add(Cropping2D(cropping=((50, 25), (0, 0))))
'''
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''

model.add(Convolution2D(24, (5, 5), strides=(2, 2),
                        activation='relu', padding='valid', W_regularizer=l2(0.001)))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid',
                        activation='relu', W_regularizer=l2(0.001)))
model.add(Convolution2D(48, (5, 5), subsample=(2, 2), padding='valid',
                        activation='relu', W_regularizer=l2(0.001)))
model.add(Convolution2D(64, (3, 3), subsample=(1, 1), padding='valid',
                        activation='relu', W_regularizer=l2(0.001)))
model.add(Convolution2D(64, (3, 3), subsample=(1, 1), padding='valid',
                        activation='relu', W_regularizer=l2(0.001)))
# model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(100, activation='relu', W_regularizer=l2(
    0.001)))
# model.add(ELU())
model.add(Dense(50, activation='relu', W_regularizer=l2(
    0.001)))
# model.add(ELU())
model.add(Dense(10, activation='relu', W_regularizer=l2(
    0.001)))
# model.add(ELU())
model.add(Dense(1))

"""
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3))
model.add(ELU())
model.add(Convolution2D(64, 3, 3))
model.add(ELU())
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(ELU())
model.add(Dense(50, activation='relu'))
model.add(ELU())
model.add(Dense(10, activation='relu'))
model.add(ELU())
model.add(Dense(1))
"""
# it's regression, not classification. hence mean square distance
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.33,
          shuffle=True, batch_size=32, nb_epoch=2)


model.save('model.h5')
print("model saved as model.h5")

print(print(model.summary()))
