
import tensorflow as tf
import config
from models.testnet import TestNet

def pretrained_model(model_name, load_weight="imagenet"):
    # TestNet
    if model_name == "TestNet":
        base_model = TestNet()
    # MobileNetV2
    if model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )
    # Xception (2017)
    if model_name == "Xception":
        base_model = tf.keras.applications.Xception(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    # EfficientNetB0~B7 (2019)
    if model_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "EfficientNetB1":
        base_model = tf.keras.applications.EfficientNetB1(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "EfficientNetB2":
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "EfficientNetB3":
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "EfficientNetB4":
        base_model = tf.keras.applications.EfficientNetB4(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "EfficientNetB5":
        base_model = tf.keras.applications.EfficientNetB5(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "EfficientNetB6":
        base_model = tf.keras.applications.EfficientNetB6(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "EfficientNetB7":
        base_model = tf.keras.applications.EfficientNetB7(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    # EfficientNetV2 B0 to B3 and S, M, L (2021)
    if model_name == "EfficientNetV2B0":
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    if model_name == "EfficientNetV2B1":
        base_model = tf.keras.applications.EfficientNetV2B1(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    if model_name == "EfficientNetV2B2":
        base_model = tf.keras.applications.EfficientNetV2B2(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    if model_name == "EfficientNetV2B3":
        base_model = tf.keras.applications.EfficientNetV2B3(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    if model_name == "EfficientNetV2S":
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    if model_name == "EfficientNetV2M":
        base_model = tf.keras.applications.EfficientNetV2M(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    if model_name == "EfficientNetV2L":
        base_model = tf.keras.applications.EfficientNetV2L(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,
        )

    # VGG series (2015)
    if model_name == "VGG16":
        base_model = tf.keras.applications.VGG16(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )

    if model_name == "VGG19":
        base_model = tf.keras.applications.VGG19(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )

    # DenseNet Series (2017)
    if model_name == "DenseNet121":
        base_model = tf.keras.applications.DenseNet121(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "DenseNet169":
        base_model = tf.keras.applications.DenseNet169(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )
    
    if model_name == "DenseNet201":
        base_model = tf.keras.applications.DenseNet201(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    # NasNet Series (2018)
    if model_name == "NASNetLarge":
        base_model = tf.keras.applications.NASNetLarge(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    if model_name == "NASNetMobile":
        base_model = tf.keras.applications.NASNetMobile(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
        )

    # InceptionV3 (2016)
    if model_name == "InceptionV3":
        base_model = tf.keras.applications.InceptionV3(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )

    # InceptionResNetV2 (2016)
    if model_name == "InceptionResNetV2":
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top=True,
            weights=load_weight,
            input_tensor=None,
            input_shape=(config.image_width,config.image_height,3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )


    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    # add logistic layer (complusory for predict classes)
    predictions = tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)


    return model
