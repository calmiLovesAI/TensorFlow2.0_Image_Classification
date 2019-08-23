from __future__ import absolute_import, division, print_function
import tensorflow as tf
from config import EPOCHS, BATCH_SIZE, model_dir
from prepare_data import get_datasets
from models.alexnet import AlexNet
from models.vgg16 import VGG16
from models.vgg19 import VGG19

def get_model():
    # model = AlexNet()
    model = VGG16()
    # model = VGG19()

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_generator, valid_generator, test_generator, \
    train_num, valid_num, test_num = get_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    callback_list = [tensorboard]

    model = get_model()
    model.summary()

    # start training
    model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // BATCH_SIZE,
                        callbacks=callback_list)

    # save the whole model
    model.save(model_dir)

