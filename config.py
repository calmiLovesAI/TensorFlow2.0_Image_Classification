# some training parameters
EPOCHS = 100
BATCH_SIZE = 128
NUM_CLASSES = 6
image_height = 128
image_width = 128
channels = 3

model_save_name = "EfficientNetB2"
model_dir = "trained_models/salmon_crop_128/"+model_save_name+"/" # = save_dir

train_dir = "/home/mirap/0_DATABASE/IMAS_Salmon/7_SalmonTest/train"
valid_dir = "/home/mirap/0_DATABASE/IMAS_Salmon/7_SalmonTest/valid"
test_dir = "/home/mirap/0_DATABASE/IMAS_Salmon/7_SalmonTest/test"
test_image_path = "/home/mirap/0_DATABASE/IMAS_Salmon/7_SalmonTest/test/5/5_108.jpg"
