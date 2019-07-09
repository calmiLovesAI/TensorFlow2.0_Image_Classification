import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(3, 3),
                               activation=tf.keras.activations.relu,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation=tf.keras.activations.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=10,
                              activation=tf.keras.activations.softmax)
    ])

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model