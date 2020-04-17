
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape,Conv2D, Conv2DTranspose, UpSampling2D,  LeakyReLU, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt


img_rows=28
img_cols=28
channel=1
input_shape = (28,28,1)

D = None
G = None
AM = None
DM = None




def make_discriminator():
    D = Sequential()
    depth = 64
    dropout = 0.4
    D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Flatten())
    D.add(Dense(1))
    D.add(Activation('sigmoid'))
    D.summary()
    return D

def make_generator():
    G = Sequential()
    dropout = 0.4
    depth = 64+64+64+64
    dim = 7
    G.add(Dense(dim*dim*depth, input_dim=100))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Reshape((dim, dim, depth)))
    G.add(Dropout(dropout))
    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(1, 5, padding='same'))
    G.add(Activation('sigmoid'))
    G.summary()
    return G



def discriminator_model():
    optimizer = RMSprop(lr=0.0002, decay=6e-8)
    DM = Sequential()
    DM.add(make_discriminator())
    DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return DM

generator = make_generator()
discriminator =  discriminator_model()

def adversarial_model():
    optimizer = RMSprop(lr=0.0001, decay=3e-8)
    AM = Sequential()
    AM.add(generator)
    AM.add(discriminator)
    AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return AM


def train(train_steps=2000, batch_size=256, save_interval=0):
    noise_input = None
    if save_interval>0:
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    for i in range(train_steps):
        images_train = x_train[np.random.randint(0,
            x_train.shape[0], size=batch_size), :, :, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        images_fake = generator.predict(noise)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        d_loss = discriminator.train_on_batch(x, y)

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        a_loss = adversarial.train_on_batch(noise, y)
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)
        if save_interval>0:
            if (i+1)%save_interval==0:
                plot_images(save2file=True, samples=noise_input.shape[0],\
                    noise=noise_input, step=(i+1))

def plot_images(save2file=False, fake=True, samples=16, noise=None, step=0):
    filename = 'mnist.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "mnist_%d.png" % step
        images = generator.predict(noise)
    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [img_rows, img_cols])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()




(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train,[-1, x_train.shape[1], x_train.shape[1], 1]).astype(np.float32)/255.

adversarial = adversarial_model()



if __name__ == '__main__':
   train(train_steps=10000, batch_size=256, save_interval=500)
   plot_images(fake=True)
   plot_images(fake=False, save2file=True)
