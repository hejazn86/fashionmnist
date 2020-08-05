# Fashionmnist
## The porpuse of this code:
* Hypermeters-tunning and examining the effect of each parameter of the network(learning rate, kernel_size, the number of filters, number of layers,number of epochs, batch_size, ...) 
* observe how the convolutions work
* how to overcome overfitting 
, ...


## Loading and preparing the data

The **Fashion_mnist** dataset, contains *70000* labeled images of 10 type of fashion_related objects like: different kinds of clothes, shoes, etc. These images are split into two datasets: **Train** and **Test**
The *Train* data comprises *60000* images of this dataset and their labels, and used to train the network, and the *Test* data contains *10000* images of this dataset and is used to evaluate the network performance.

First we import the libraries we need, so we import `tensorflow`, `numpy`, and `matplotlib.pyplot`.
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```
Then we are going to load the dataset and split it to train and test data using  the following cod:
`(x_train, y_train), (x_test, y_test) = tf.keras. datasets.fashion_mnist.load_data()`  

