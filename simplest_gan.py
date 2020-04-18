import os
import math

import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop


# global stuff
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train,[-1, x_train.shape[1], x_train.shape[1], 1]).astype(np.float32)
x_train /= 255

steps = 10000
batch_size=256

# rows, columns, channels
img_shape = (28,28,1)
compile_kwargs = {'loss':'binary_crossentropy', 'metrics': ['accuracy']}

def dropout():
    return Dropout(0.4)

def sigmoid():
    return Activation('sigmoid')


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


# discriminator params
strides = [2]*2 + [1]
disc_depth = 64

def leakyrelu():
    return LeakyReLU(alpha=0.2)

# generator params
gen_depth = 256
dim = 7

def batchnorm():
    return BatchNormalization(momentum=0.9)

def relu():
    return Activation('relu')


# build discriminator
discriminator = Sequential()

discriminator.add(Conv2D(disc_depth, 5, strides=2, input_shape = img_shape, padding='same'))
discriminator.add(leakyrelu())
discriminator.add(dropout())

for i in range(len(strides)):
    discriminator.add(Conv2D(disc_depth * 2 ** (i+1), 5, strides = strides[i], padding='same'))
    discriminator.add(leakyrelu())
    discriminator.add(dropout())

discriminator.add(Flatten())
discriminator.add(Dense(1))
discriminator.add(sigmoid())

print(discriminator.summary())

discriminator.compile(optimizer=RMSprop(learning_rate=0.0002, decay=6e-8), **compile_kwargs)



# build Generator
generator = Sequential()

generator.add(Dense(dim * dim * gen_depth, input_dim = 100))
generator.add(batchnorm())
generator.add(relu())
generator.add(Reshape((dim, dim, gen_depth)))
generator.add(dropout())

for i in range(1,4):
    if i != 3:
        generator.add(UpSampling2D())
    generator.add(Conv2DTranspose(gen_depth // (2 ** i), 5, padding = 'same'))
    generator.add(batchnorm())
    generator.add(relu())

generator.add(Conv2DTranspose(1, 5, padding='same'))
generator.add(sigmoid())
print(generator.summary())


# build GAN
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
# compile things
#gan.layers[-1].trainable=False
gan.compile(optimizer=RMSprop(learning_rate=1e-4, decay=3e-8), **compile_kwargs)

print(gan.summary())

# train the model
# make space for plots
if not os.path.exists("images/"):
    os.mkdir("images/")


img_noise = np.random.uniform(-1.0,1.0, size=[64, 100])

for s in range(steps):
    images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    images_fake = generator.predict(noise)
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2*batch_size, 1])
    y[batch_size:, :] = 0
    d_loss =discriminator.train_on_batch(x, y)
    y = np.ones([batch_size, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    a_loss = gan.train_on_batch(noise, y)
    log_mesg = "%d: [D loss: %f, acc: %f]" % (s, d_loss[0], d_loss[1])
    log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
    print(log_mesg)

    if (s+1) %250 == 0:
        fname = "images/mnist_{0:05d}.png".format(s+1)
        images = combine_images(generator.predict(img_noise))
        images *= 255.
        Image.fromarray(images.astype(np.uint8)).save(fname)




