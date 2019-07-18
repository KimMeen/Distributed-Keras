#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:47:40 2019

Distributed training based on Keras

@author: Ming Jin
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import multi_gpu_model

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_data(input_file):
    d = unpickle(input_file)
    x = d['data']
    y = d['labels']
    x = np.dstack((x[:, :4096], x[:, 4096:8192], x[:, 8192:]))
    x = x.reshape((x.shape[0], 64, 64, 3))
#    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
#    x = x.reshape((x.shape[0], 32, 32, 3))
    return x, y

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def split_dataset(features, labels, training = 0.8, validation = 0.3):
    rnd_indices = np.random.rand(len(labels)) < training
    train_x = features[rnd_indices]
    train_y = labels[rnd_indices]
    remain_x = features[~rnd_indices]
    remain_y = labels[~rnd_indices]
    
    rnd_indices2 = np.random.rand(len(remain_y)) < validation
    val_x = remain_x[rnd_indices2]
    val_y = remain_y[rnd_indices2]
    test_x = remain_x[~rnd_indices2]
    test_y = remain_y[~rnd_indices2]
    return train_x, train_y, val_x, val_y, test_x, test_y

# define hyper parameters

_GPUs = 2
_LR = 0.01
_EPOCH = 200
_BATCH_SIZE = 128

# loading data from binary files
X = []
Y = []
directory = '/home/tpc2/Downloads/64*64/training_data'

for i in range(1):
    i = i + 1 
    x, y = load_data(directory + '/train_data_batch_%d' % i)
    X.extend(x)
    Y.extend(y)
    print('%d out of 10 files' % i)
    
X = np.array(X)
Y = np.array(Y)
Y = dense_to_one_hot(Y, 1000)

print(X.shape)
print(Y.shape)

train_x, train_y, val_x, val_y, test_x, test_y = split_dataset(X, Y)

# preparing data generator

train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_generator = train_datagen.flow(train_x, train_y, batch_size=_BATCH_SIZE * _GPUs)
validation_generator = test_datagen.flow(val_x, val_y, batch_size=_BATCH_SIZE * _GPUs)
test_generator = test_datagen.flow(test_x, test_y, batch_size=_BATCH_SIZE * _GPUs)


#with tf.device('/cpu:0'):
#    model = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(32,32,3), pooling=None, classes=1000)
model = ResNet50(weights=None, input_shape=(64,64,3))
opt = tf.keras.optimizers.SGD(lr = _LR, momentum = 0.9)

para_model = multi_gpu_model(model, gpus = _GPUs, cpu_merge = True, cpu_relocation = False)
para_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

callbacks = [
    # Reduce the learning rate if training plateaues.
    tf.keras.callbacks.ReduceLROnPlateau(patience=60, verbose = 1),
   
    # Early stopping
#    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, verbose = 1, mode = 'auto', baseline = None),
    
    # Tensorboard
    tf.keras.callbacks.TensorBoard(log_dir='./keras_logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=False)
    ]

para_model.fit_generator(
            train_generator,
            epochs =_EPOCH,
            callbacks = callbacks,
            validation_data = validation_generator)

print('\n')
score = para_model.evaluate_generator(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])