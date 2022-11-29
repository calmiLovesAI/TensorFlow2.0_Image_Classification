import tensorflow as tf
import config
from prepare_data import get_datasets

train_generator, valid_generator, test_generator, \
train_num, valid_num, test_num= get_datasets()


# Load the model
new_model = tf.keras.models.load_model(config.model_dir+config.model_save_name+".h5")
# Get the accuracy on the test set
loss, acc, auc, precision, recall  = new_model.evaluate(test_generator,
                                         batch_size=config.BATCH_SIZE,
                                         steps=test_num // config.BATCH_SIZE)
print("result of ",config.model_dir)
print("The accuracy on test set is: {:6.3f}%".format(acc*100))
print("The auc on test set is: {:6.3f}%".format(auc*100))
print("The precision on test set is: {:6.3f}%".format(precision*100))
print("The recall on test set is: {:6.3f}%".format(recall*100))
