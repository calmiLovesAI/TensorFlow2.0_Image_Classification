import tensorflow as tf
from sklearn.model_selection import train_test_split


def get_datasets():
    # Download the dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Divide the training dataset into training dataset and validation dataset
    train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels,
                                                                              test_size=0.2,
                                                                              random_state=1)

    train_images = train_images.reshape((-1, 28, 28, 1))
    valid_images = valid_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    num_of_train_images = train_images.shape[0]

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    valid_images = valid_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, \
           valid_images, valid_labels, \
           test_images, test_labels, \
           num_of_train_images