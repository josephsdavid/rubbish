import numpy as np
import time
import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

disable_eager_execution()

import matplotlib.pyplot as plt

