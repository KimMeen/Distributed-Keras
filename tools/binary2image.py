#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:59:19 2019

Ming Jin

Transform data to raw images (unfinish)
"""

import numpy as np
from PIL import Image
import os
import pickle
from argparse import ArgumentParser

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_data(input_file):
    d = unpickle(input_file)
    x = d['data']
    y = d['labels']
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    return x, y

def transfrom2image(x):
    for i in range(x.shape[0]):
        blank_image = None
        blank_image = Image.new('RGB', (32, 32))
        blank_image.paste(Image.fromarray(x[i]))
        blank_image.save('./training/Image_%d.png' % i)
        
if __name__ == '__main__':
    directory = '/mnt/data/ImageNet/32*32/training_data/'
    
    for i in range(1000):
        i = i + 1
        os.mkdir(directory + 'i')

    for i in range(10):
        i = i + 1
        x, y = load_data(directory + '/train_data_batch_%d' % i)
        transfrom2image(x)