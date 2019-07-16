#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:47:40 2019

Distributed training based on Horovod

@author: Ming Jin
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import horovod.tensorflow.keras as hvd
from tensorflow.keras import backend as K


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
_LR = 0.01
_EPOCH = 200
_BATCH_SIZE = 128

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config = config))

# loading data from binary files
X = []
Y = []
directory = '/home/tpc2/Downloads/64*64/training_data'

for i in range(1):
    i = i + 1 
    x, y = load_data(directory + '/train_data_batch_%d' % i)
    X.extend(x)
    Y.extend(y)
    if hvd.rank() == 0:
        print('%d out of 10 files' % i)
    
X = np.array(X)
Y = np.array(Y)
Y = dense_to_one_hot(Y, 1000)

if hvd.rank() == 0:
    print(X.shape)
    print(Y.shape)

train_x, train_y, val_x, val_y, test_x, test_y = split_dataset(X, Y)

# Determine how many batches are there in train and test sets
train_batches = len(train_x) // _BATCH_SIZE
val_batches = len(val_x) // _BATCH_SIZE

# preparing data generator

train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_generator = train_datagen.flow(train_x, train_y, batch_size=_BATCH_SIZE)
validation_generator = test_datagen.flow(val_x, val_y, batch_size=_BATCH_SIZE)
test_generator = test_datagen.flow(test_x, test_y, batch_size=_BATCH_SIZE)

#model = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(32,32,3), pooling=None, classes=1000)
model = ResNet50(weights=None, input_shape=(64,64,3))

opt = tf.keras.optimizers.SGD(lr = _LR * hvd.size(), momentum = 0.9)
opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)
    
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    
    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
#    hvd.callbacks.MetricAverageCallback(),
    
    # Horovod: set up warmup epochs before adjust the learning rate
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=10, verbose=1),
    
    # Reduce the learning rate if training plateaues.
    tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose = 1)
   
    # Early stopping
#    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, verbose = 1, mode = 'auto', baseline = None),
    
    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
#    hvd.callbacks.LearningRateScheduleCallback(start_epoch=10, end_epoch=30, multiplier=1.),
#    hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
#    hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
#    hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
    ]

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='./horovod_logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=False))

model.fit_generator(
        train_generator,
        steps_per_epoch = train_batches // hvd.size(),
        epochs =_EPOCH,
        callbacks = callbacks,
        workers = 8,
        validation_data = validation_generator,
        validation_steps = 3 * val_batches // hvd.size())

print('\n')
score = hvd.allreduce(model.evaluate_generator(test_generator, workers = 8))
print('Test loss:', score[0])
print('Test accuracy:', score[1])