# Fashionmnist
## The porpuse of this code:
* Hypermeters-tunning and examining the effect of each parameter of the network(learning rate, kernel_size, the number of filters, number of layers,number of epochs, batch_size, ...) 
* observe how the convolutions work
* how to overcome overfitting 
, ...


## Loading and preparing the data

The **Fashion_mnist** dataset, contains *70000* labeled images of 10 type of fashion_related objects like: different kinds of clothes, shoes, etc. These images are split into two datasets: **Train** and **Test**
The *Train* data comprises *60000* images of this dataset and their labels, and used to train the network, and the *Test* data contains *10000* images of this dataset and is used to evaluate the network performance.


SO first we import the libraries we need, so we import `tensorflow`, `numpy`, and `matplotlib.pyplot`.
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```
Then we are going to load the dataset using the following cod:

`(x_train, y_train), (x_test, y_test) = tf.keras. datasets.fashion_mnist.load_data()` 

After loading the data, we need to prepare the data to fit the network, therefore, we expand the *dimension* of the data using one of this codes:
```
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
```
```
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
```
Image data are represented as *multi-dimensional arrays*, each image in this dataset is a *two-dimensional arrays* representing the width and the height of the image. since we are going to feed this images to **convolutional layers**, we should specify their channels as well, so we `reshape` the images to add another dimension to the them. This dimension represents the channels of the images: *3* for *coloured* pictures, and *1* for *grayscale* pictures.

Since the dataset contains grayscale images, we use *1*.

Next, we need to normalize the arrayes, in order to do this, we divide every element of the array by the largest one. For images, arrays takes the values from *0* to *255*, so each number of the array is divided by *255*. But, first we need to change the type of the data to be able to divide them by *255*, because the operation will return float numbers between *0* and *255*. Images arrays regularly takes *Ineger numbers*, we use `x_train = x_train.astype("float32")` to change the type of train data from _integer_ to _float_, and do the same thing for text data. The code bellow shows this operation: 

```
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= np.argmax(x_train)
x_test /= np.argmax(x_test)
```

The labels in the dataset are integer numbers between *(0, 10)*, each of them represents a class of the images. To feed these labels to the network, they should be converted to arrays, one-dimensional array with *10 columns*, each column represents a class of data. Nine of the ten elemnts of the array will be *0* and just one of them will be *1* for each class, For example, the in array representing the *label 3* the *column 3* takes the value *1* :
`[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`

The following code perform this action both for `train_labels` and `test_labels`:

```
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
```



