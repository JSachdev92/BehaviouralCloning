import os
import matplotlib.image as mpimg
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


#cd /home/workspace/CarND-Behavioral-Cloning-P3
samples = []

#process data from csv
data_path = './Drive_Data/'
csv_path_filename = data_path + 'driving_log.csv'
images_path = data_path + 'IMG/'
with open(csv_path_filename) as csvfile:
    reader = csv.reader(csvfile)
    next(reader,'None')
    for line in reader:
        samples.append(line)

               
print('Total number of frames = ', len(samples))

recorded_images = []
measurement_angles = []

#for sample in samples:
#   source_path = sample[0]   
#    filename = source_path.split('/')[-1]  
#    image = cv2.imread(images_path + filename)
#    recorded_images.append(image)
#    measurement_angle = float(sample[3])
#    measurement_angles.append(measurement_angle)
        
#X_train = np.array (recorded_images)
#y_train = np.array(measurement_angles)

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#implement NVidia Deep Neural Network for Autonomous Vehicles
#Add Dropouts to prevent overfitting

model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,25),(0,0))))

model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
#model.add(Dropout(0.4, noise_shape=None, seed=None))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())

model.add(Dropout(0.5, noise_shape=None, seed=None))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#model.summary()


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]  
                for index in ["center", "left", "right"]:
                    filename = filename.split('_')  
                    filename[0]=index
                    filename='_'.join(filename)
                    image = cv2.imread(images_path + filename)
                    images.append(image)
                    if index == "center":
                        images.append(cv2.flip(image, 1))
                
                #Corecction of 0.2 for left and right images
                correction = 0.2
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(-1.0 * angle)
                angles.append(angle + correction)
                angles.append(angle - correction)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Splitting the train and test samples and creating batches using the generator
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


batch_size = 100

print('Number of Training Images:', len(train_samples))
print('Number of Validation Images:', len(validation_samples))

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.compile(loss='mse', optimizer='adam')

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

epochs = 15

history_object = model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=(len(validation_samples)/batch_size), epochs=epochs, callbacks=callbacks_list, verbose=1)

#save only best model to help prevent overfitting if validation is not improving
model.save('model_last_epoch.h5')
model.load_weights("weights.best.hdf5")
model.save('model.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
