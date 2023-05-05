# TensorFlow2.0_Image_Classification(include AlexNet and VGGNet)
This project uses TensorFlow2.0 for image classification tasks.

## How to use
### Requirements
+ **Python 3.x** (My Python version is 3.8.0)<br/>
+ **TensorFlow version: 2.11.0 <br/> 
+ The file directory of the dataset should look like this: <br/>
```
${dataset_root}
|——train
|   |——class_dir_0
|   |  |——image_1.jpg
|   |  |——image_2.jpg
|   |  |——image_3.jpg
|   |  ...
|   |——class_dir_1
|   |——class_dir_2
|   |——class_dir_3
|——valid
|   |——class_dir_0
|   |——class_dir_1
|   |——class_dir_2
|   |——class_dir_3
|——test
    |——class_dir_0
    |——class_dir_1
    |——class_dir_2
    |——class_dir_3
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
1. AlexNet : http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
2. VGG : https://arxiv.org/abs/1409.1556
3. Keras : https://keras.io/api/applications/
