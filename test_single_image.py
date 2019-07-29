import tensorflow as tf
import config
import numpy as np


def test_single_image(img_dir):
    img_raw = tf.io.read_file(img_dir)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=config.channels)
    img_tensor = tf.image.resize(img_tensor, [config.image_height, config.image_width])

    img_numpy = img_tensor.numpy()
    img_numpy = (np.expand_dims(img_numpy, 0))
    img_tensor = tf.convert_to_tensor(img_numpy, tf.float32)
    # print(img_tensor.shape)
    img = img_tensor / 255.0
    prob = model(img)
    # print(prob)
    classification = np.argmax(prob)

    return classification


if __name__ == '__main__':
    model = tf.keras.models.load_model(config.model_dir)
    classification = test_single_image(config.test_image_path)
    print(classification)