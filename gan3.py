#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tensorflow.keras.datasets import mnist
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dropout

import math
import numpy as np
import sys

def combine_images(generated_images):
    total,width,height = generated_images.shape[:-1]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image

def show_progress(e,i,g0,d0,g1,d1):
    sys.stdout.write("\repoch: %d, batch: %d, g_loss: %f, d_loss: %f, g_accuracy: %f, d_accuracy: %f" % (e,i,g0,d0,g1,d1))
    sys.stdout.flush()


def generator(input_dim=100,units=1024,activation='relu'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    print(model.summary())
    return model

def discriminator(input_shape=(28, 28, 1),nb_filter=64):
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same',
                            input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(2*nb_filter, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4*nb_filter))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model



BATCH_SIZE = 256
NUM_EPOCH = 50
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'

def train():
    (X_train, y_train), (_, _) = mnist.load_data()
    # normalize images
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    # build GAN
    g = generator()
    d = discriminator()

    opt = Adam(lr=LR,beta_1=B1)
    d.trainable = True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)
    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR,beta_1=B1)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    # create directory
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)

    print("-------------------")
    print("Total epoch:", NUM_EPOCH, "Number of batches:", num_batches)
    print("-------------------")
    z_pred = np.array([np.random.uniform(-1,1,100) for _ in range(49)])
    y_g = [1]*BATCH_SIZE
    y_d_true = [1]*BATCH_SIZE
    y_d_gen = [0]*BATCH_SIZE
    for epoch in list(map(lambda x: x+1,range(NUM_EPOCH))):
        for index in range(num_batches):
            X_d_true = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            X_g = np.array([np.random.normal(0,0.5,100) for _ in range(BATCH_SIZE)])
            X_d_gen = g.predict(X_g, verbose=0)

            # train discriminator
            d_loss = d.train_on_batch(X_d_true, y_d_true)
            d_loss = d.train_on_batch(X_d_gen, y_d_gen)
            # train generator
            g_loss = dcgan.train_on_batch(X_g, y_g)
            show_progress(epoch,index,g_loss[0],d_loss[0],g_loss[1],d_loss[1])

        # save generated images
        image = combine_images(g.predict(z_pred))
        image = image*127.5 + 127.5
        Image.fromarray(image.astype(np.uint8))\
            .save(GENERATED_IMAGE_PATH+"%03depoch.png" % (epoch))
        print()
        # save models
        g.save(GENERATED_MODEL_PATH+'dcgan_generator.h5')
        d.save(GENERATED_MODEL_PATH+'dcgan_discriminator.h5')

if __name__ == '__main__':
    train()
