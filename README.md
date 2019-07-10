# TensorFlow2.0_Image_Classification
The classification of images with TensorFlow_2.0

## How to use
### Requirements
+ **Python 3.x** (My Python version is 3.6.8)<br/>
+ **TensorFlow version: 2.0.0-beta1**<br/> 
+ The file directory of the dataset should look like this: <br/>
```
${dataset_root}
|——train
|   |——class_name_0
|   |——class_name_1
|   |——class_name_2
|   |——class_name_3
|——valid
|   |——class_name_0
|   |——class_name_1
|   |——class_name_2
|   |——class_name_3
|——test
    |——class_name_0
    |——class_name_1
    |——class_name_2
    |——class_name_3
```

### Train
Run the script
```
python train.py
```
to train the network on your image dataset, the final model will be stored. You can also change the corresponding training parameters in the `config.py`.<br/>

### Evaluate
To evaluate the model's performance on the test dataset, you can run `evaluate.py`.<br/>

The structure of the network is defined in `model_definition.py`, you can change the network structure to whatever you like.<br/>

## References
Part of the code is referenced from the offcial tutorial https://tensorflow.google.cn/beta/tutorials/images/intro_to_cnns.
