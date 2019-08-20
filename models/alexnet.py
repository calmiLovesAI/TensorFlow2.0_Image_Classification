import tensorflow as tf
from config import NUM_CLASSES

def AlexNet():
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="valid",
                               activation=tf.keras.activations.relu,
                               input_shape=(227, 227, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="valid"),
        tf.keras.layers.BatchNormalization(),
        # layer 2
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 3
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 4
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 5
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 6
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 7
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 8
        tf.keras.layers.Dense(units=NUM_CLASSES,
                              activation=tf.keras.activations.softmax)
    ])

    return model