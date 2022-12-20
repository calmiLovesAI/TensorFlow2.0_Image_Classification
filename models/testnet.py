import tensorflow as tf
from config import NUM_CLASSES, image_width, image_height, channels


def TestNet():
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=4,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same",
                               activation=tf.keras.activations.relu,
                               input_shape=(image_height, image_width, channels)),
        #tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=2,
                                  padding="valid"),
        tf.keras.layers.Conv2D(filters=4,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=2,
                                  padding="valid"),
        # tf.keras.layers.BatchNormalization(),
        # layer 2
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=512,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=512,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=512,
                              activation=tf.keras.activations.relu),
        # tf.keras.layers.Dropout(rate=0.5),
    ])

    return model