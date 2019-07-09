# TensorFlow2.0-MNIST
The classification of images from MNIST datasets with TensorFlow_2.0

## How to use
### Train
Run the script
```
python train.py
```
to train the network on the MNIST dataset, the final model will be stored. You can also change the corresponding training parameters in the `config.py`.<br/>

### Evaluate
To evaluate the model's performance on the test dataset, you can run `evaluate.py`.<br/>

The structure of the network is defined in `model_difinition.py`, you can change the network structure to whatever you like.<br/>

## References
Part of the code is referenced from the offcial tutorial https://tensorflow.google.cn/beta/tutorials/images/intro_to_cnns.
