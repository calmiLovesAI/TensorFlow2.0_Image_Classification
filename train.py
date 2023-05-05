from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import config
from test_single_image import test_single_image
from evaluate import eval_model
from prepare_data import get_datasets
from models.pretrained_models import pretrained_model

# USAGE: python train.py 
#((before run! please set config.py file and line 26 of train.py))

available_models=["Xception","TestNet",
                  "EfficientNetB0", "EfficientNetB1", "EfficientNetB2",
                  "EfficientNetB3", "EfficientNetB4", "EfficientNetB5",
                  "EfficientNetB6", "EfficientNetB7",
                  "EfficientNetV2B0", "EfficientNetV2B1",
                  "EfficientNetV2B2", "EfficientNetV2B3",
                  "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L",
                  "VGG16","VGG19",
                  "DenseNet121", "DenseNet169", "DenseNet201",
                  "NASNetLarge","NASNetMobile",
                  "InceptionV3","InceptionResNetV2"
                  ]

def get_model():
    model = pretrained_model(model_name="TestNet",
                            load_weight=None)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  metrics=['accuracy', # add more metrics if you want
                            tf.keras.metrics.AUC(),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            ])
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

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.model_dir,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=20,
        restore_best_weights=True
    )
    callback_list = [tensorboard, model_checkpoint_callback, early_stop_callback]

    model = get_model()
    #model.summary()

    # start training
    history = model.fit(train_generator,
                        epochs=config.EPOCHS,
                        steps_per_epoch=train_num // config.BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // config.BATCH_SIZE,
                        callbacks=callback_list)

    # save the whole model
    model.save(config.model_dir+config.model_save_name+".h5")
    
    #write histry
    hist_df = pd.DataFrame(history.history)
    with open(config.model_dir+"train_history.csv", mode='w') as f:
        hist_df.to_csv(f)

    # Evaluation
    eval_model(model)

    # detect for samples
    test_single_image(config.test_image_path, model)

    print("end of training!!!")