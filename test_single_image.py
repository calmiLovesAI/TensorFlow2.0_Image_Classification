import tensorflow as tf
import config
import numpy as np
import os

def test_single_image(img_dir, model):
    img_raw = tf.io.read_file(img_dir)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=config.channels)
    img_tensor = tf.image.resize(img_tensor, [config.image_height, config.image_width])

    img_numpy = img_tensor.numpy()
    img_numpy = (np.expand_dims(img_numpy, 0))
    img_tensor = tf.convert_to_tensor(img_numpy, tf.float32)
    # print(img_tensor.shape)
    img = img_tensor / 255.0
    prob = model(tf.image.resize(img,[config.image_width,config.image_height]))
    # print(prob)
    classification = np.argmax(prob)

    # print result
    head, filename = os.path.split(img_dir)
    result_s = "[result] "+ filename + " : " + str(classification)
    print(result_s)

    # save result
    f = open(config.model_dir+"test_result.txt", "w")
    f.write(result_s)
    f.close()

    return classification


if __name__ == '__main__':
    model = tf.keras.models.load_model(config.model_dir+config.model_save_name+".h5")
    test_image = config.test_image_path
    classification = test_single_image(test_image, model)
    