import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Lambda, Cropping2D, Flatten, Dropout
from keras.layers.convolutional import Convolution2D

samples = []

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # ignore csv header
    next(reader)

    for row in reader:

        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # img paths to 3 different camera angles
        img_center = '../data/IMG/'+row[0].split('/')[-1]
        img_left = '../data/IMG/'+row[1].split('/')[-1]
        img_right = '../data/IMG/'+row[2].split('/')[-1]

        # append the 3 images and steering angles to image samples
        samples.append([img_center, steering_center])
        samples.append([img_left, steering_left])
        samples.append([img_right, steering_right])


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    """
    Provides the Tensorflow fit_generator() function with batches of image samples
    to reduce memory usage.

    :param samples: list
    :param batch_size: int
    :yield: tuple(numpy array, numpy array)
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        # sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                name = batch_sample[0]
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image, 1))
                angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.4))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*2,
                    nb_epoch=5,
                    verbose=1)

model.save('model.h5')
