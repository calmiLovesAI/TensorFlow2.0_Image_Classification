import tensorflow as tf
import config
from prepare_data import get_datasets

train_generator, valid_generator, test_generator, \
train_num, valid_num, test_num= get_datasets()


# Load the model
new_model = tf.keras.models.load_model(config.model_dir)
# Get the accuracy on the test set
loss, acc = new_model.evaluate_generator(test_generator,
                                         steps=test_num // config.BATCH_SIZE)
print("The accuracy on test set is: {:6.3f}%".format(acc*100))