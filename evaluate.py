import tensorflow as tf
from config import model_dir
from prepare_data import get_datasets


train_images, train_labels, \
valid_images, valid_labels, \
test_images, test_labels, \
num_of_train_images = get_datasets()


# Load the model
new_model = tf.keras.models.load_model(model_dir)
# Get the accuracy on the test set
loss, acc = new_model.evaluate(test_images, test_labels)
print("The accuracy on test set is: {:6.3f}%".format(acc*100))