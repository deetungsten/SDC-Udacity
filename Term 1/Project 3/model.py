#Importing all the required libraries
from keras.models import Sequential
from keras.layers import Convolution2D,  Dense, Flatten, Activation, Dropout
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
from keras.optimizers import adam
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import misc
from random import random

# Defining the neural network architecture
def baseline_model():
    # create model
    nb_input = 1

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    #model.add(Dense(1164,activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.6))
    model.add(Dense(50, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    #model.add(Dropout(0.3))
    model.add(Dense(nb_input))
    # Compile model
    #adamfunc = adam(lr=0.0018, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def data_gen(df):

    X_train = []
    y_train = []

    #ITERATING THROUGH THE CSV FILE TO LOAD ALL THE IMAGES INTO THE RAM TO USE LATER FOR BEHAVIOR LEARNING
    for index, row in df.iterrows():
        img_center = misc.imread(df[0][index])
        #img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2GRAY)

        img_left = misc.imread(df[1][index])
        img_right = misc.imread(df[2][index])


        steering_center = float(df[3][index])

        plt.show()
        correction = 0.15  # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        throttle = float(df[4][index])
        break_value = float(df[5][index])
        speed = float(df[6][index])


        X_train.append(img_center)
        y_train.append([steering_center])

        X_train.append(img_left)
        y_train.append([steering_left])

        X_train.append(img_right)
        y_train.append([steering_right])

        draw = random()

        #OVER 4 GB GIVES ME PROBLEMS WITH PICKLE (WHICH FREEZES MY COMPUTER).
        #RANDOMLY FLIP 30% OF THE IMAGES TO CREATE ARTIFICALLY AUGMENTED DATA.

        if draw <= 0.3:
            X_train.append(np.fliplr(img_left))
            y_train.append([-steering_left])

            X_train.append(np.fliplr(img_right))
            y_train.append([-steering_right])


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train

def data_import():
    csv_file = 'driving_log.csv'
    df = pd.read_csv(csv_file, header=None)

    #IF THE PICKLE DATA EXIST, LOAD IT TO SAVE TIME
    #IF DOES NOT EXIST, LOAD THE IMAGES AND PACKAGE INTO A PICKLE FILE
    try:
        X_train = pickle.load(open("X_train.p", "rb"))
        y_train = pickle.load(open("y_train.p", "rb"))
        print('data exists')

        image_shape = X_train.shape
        output_shape = y_train.shape

        print("Image data shape =", image_shape)
        print()
        print("Output data shape =", output_shape)
        print ()

    except (OSError, IOError) as e:


        start = time.time()
        print("data generation START")
        X_train, y_train = data_gen(df)

        image_shape = X_train.shape
        output_shape = y_train.shape

        print("Image data shape =", image_shape)
        print()
        print("Output data shape =", output_shape)
        print ()

        end = time.time()
        print("data generation COMPLETE")
        print(end - start)

        pickle.dump(X_train, open( "X_train.p", "wb" ))
        pickle.dump(y_train, open( "y_train.p", "wb" ))
        print('NOT EXIST- DATA SAVED')
    return X_train, y_train




#LOADS THE DATA
X_train, y_train = data_import()


#DEFINE SOME OF THE TOP LEVEL HYPERPARAMETERS
batch_size = 8
nb_epoch = 3

#IMPORT THE NEURAL NETWORK MODEL THAT I DEFINED
model = baseline_model()

#SHUFFLING THE DATA
X_train, y_train = shuffle(X_train,y_train)

#SPLITTING THE DATA FOR VALIDATION AND TRAINING (20%)
X_train, X_test, y_train, y_test = train_test_split(X_train, ytest_train, test_size=0.2, random_state=10)


#DISPLAYING SOME METRICS TO MAKE SURE THE SHAPES ARE CORRECT
print("AFTER SPLIT, TRAIN SIZE =", X_train.shape)
print()
print("AFTER SPLIT, TEST SIZE =", X_test.shape)
print()


print('TRAINING')

# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     horizontal_flip=True)
#datagen.fit(X_train)

#EARLY STOPPING TO PREVENT OVERFITTING, HOWEVER WITH 2 EPOCHS THIS IS RENDERED USELESS
early_stopping = EarlyStopping(monitor='val_loss', patience=2,mode='auto')

#I TRIED USING KERAS TO CREATE THE AUGMENTED ARTIFICAL DATA. WORKED BETTER MANUALLY FOR THIS PROJECT
#model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch=len(X_train), nb_epoch=nb_epoch,validation_data=(X_test, y_test),callbacks=[early_stopping])

#FIT THE DATA
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(X_test, y_test),callbacks=[early_stopping],shuffle= True)


#SAVE THE DATA
model.save('model.h5')


print('COMPLETE')