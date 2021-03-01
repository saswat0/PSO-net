# Fitness without optimization 
import keras
from keras import Sequential
from keras.layers import *
import keras_metrics
from keras.utils import multi_gpu_model

import tensorflow as tf
import numpy as np

def create_base():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(92, 196, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1500, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1251, activation='softmax'))
    return model

def create_model_single():
    opt = keras.optimizers.SGD(lr=0.01)
    model = create_base()
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])    
    return model

def create_model_multi():
    opt = keras.optimizers.SGD(lr=0.01)
    model = create_base()
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    return parallel_model