from __future__ import absolute_import, division, print_function
import tensorflow as tf
from model_definition import create_model
from config import EPOCHS, BATCH_SIZE, model_dir
from prepare_data import get_datasets

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

train_images, train_labels, \
valid_images, valid_labels, \
test_images, test_labels, \
num_of_train_images = get_datasets()


# start training
model = create_model()
model.fit(x=train_images,
          y=train_labels,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          steps_per_epoch=num_of_train_images // BATCH_SIZE,
          validation_data=[valid_images, valid_labels])

# save the whole model
model.save(model_dir)

