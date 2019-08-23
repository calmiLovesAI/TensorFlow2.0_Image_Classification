import tensorflow as tf
from config import *


def VGG16():
    model = tf.keras.Sequential()
    # 1
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu,
                                     input_shape=(image_height, image_width, channels)))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))

    # 2
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))

    # 3
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))

    # 4
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))

    # 5
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096,
                                    activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=4096,
                                    activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=NUM_CLASSES,
                                    activation=tf.keras.activations.softmax))

    return model