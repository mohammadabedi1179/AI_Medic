import tensorflow as tf
import numpy as np

lr_shcedules = tf.keras.callbacks.LearningRateScheduler(lambda epoch : 1(np.exp(-8)))