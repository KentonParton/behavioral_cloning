import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Lambda, Cropping2D, Flatten
from keras.layers.convolutional import Convolution2D


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples[1:]:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image, 1))
                    angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            print(len(X_train))
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Pre-process incoming data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: x/127.5 - 1.,
#                  input_shape=(ch, row, col),
#                  output_shape=(ch, row, col)))

model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3)

model.save('model.h5')