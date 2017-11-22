# Convolutional Neural Networks

## Project 1

We use the image data in MNIST from keras.datasets.Every MNIST data point has two parts: an image of a handwritten digit and a corresponding label.Both the training set and test set contain images and their corresponding labels; for example the training images are mnist.train.images and the training labels are mnist.train.labels.Each image is 28 pixels by 28 pixels. 
  
  We build a sequential model using keras with paramaters as
  
  - filters=64,kernel_size=[2,2],strides= 1, activation='relu', input_shape=(28,28,1)))
  
  - Dropout as a method for regularizing our model in order to prevent overfitting. 
  
  - MaxPooling2D as a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across 
     the previous layer and taking the max of the 4 values in the 2x2 filter.
     
   - loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] 
     
References: 
Reference: [https://elitedatascience.com/keras-tutorial-deep-learning-in-python]




Implementation of Convolutional Neural Networks in R using "mxnet "& in Python using "keras"

*Data set Description for implementation in R:*

I am using the Olivetti faces dataset. This dataset is a collection of 64×64 pixel 0-256 greyscale images.

The dataset contains a total of 400 images of 40 subjects with just 10 samples for each subject.

The dataset is credited to AT&T Laboratories Cambridge.

*Data set Description for implementation in Python:*

Every MNIST data point has two parts: an image of a handwritten digit and a corresponding label.

Both the training set and test set contain images and their corresponding labels; for example the training images are mnist.train.images and the training labels are mnist.train.labels.

Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:

References: 
Reference: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
