import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Lambda, Cropping2D, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2


def flip_img(img):
    """
    Flips an img.
    Uses: Augments training data and prevents overfitting on a track which has more left/right
    turns.
    :param img:
    :return: Flipped image data
    """
    return cv2.flip(img, 1)


def invert_angle(angle):
    """
    Inverts the steering angle for the flip_img()
    :param angle:
    :return: inverted angle
    """
    return angle * -1.0


samples = []

# open csv file with image dirs and steering angles
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # ignore csv header
    next(reader)

    for row in reader:

        # center steering angle
        steering_center = float(row[3])

        # adjusted steering measurements for the side camera images
        correction = 0.2  # steering angle correction parameter
        # left steering angle
        steering_left = steering_center + correction
        # right steering angle
        steering_right = steering_center - correction

        # img paths to 3 different camera angles
        img_center = './data/IMG/'+row[0].split('/')[-1]
        img_left = './data/IMG/'+row[1].split('/')[-1]
        img_right = './data/IMG/'+row[2].split('/')[-1]

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
        # loop through batches of the entire sample set
        for offset in range(0, num_samples, batch_size):
            # batch sample
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # dir path to image
                path = batch_sample[0]

                # read in image
                image = cv2.imread(path)
                # steering angle
                angle = float(batch_sample[1])

                # append image and steering angle
                images.append(image)
                angles.append(angle)

                # append augmented data, flipped image and inverted steering angle
                images.append(flip_img(image))
                angles.append(invert_angle(angle))

            # create numpy arrays of batch data
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Connvolutional Neural Network using Keras
model = Sequential()
# Normalize data
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
# Crop top 70 pixels and bottom 20 pixels of image
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# Convolutional Layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
# Flatten points
model.add(Flatten())

# 5 Densely connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# double samples_per_epoch and nb_val_samples because of augmented data.
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*2,
                    nb_epoch=3,
                    verbose=1)

# save the created trained model
model.save('model.h5')
