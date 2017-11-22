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

## Project 2

We use the Olivetti faces dataset. This dataset is a collection of 64×64 pixel 0-256 greyscale images.The dataset contains a total of 400 images of 40 subjects with just 10 samples for each subject.

The dataset is credited to AT&T Laboratories Cambridge.

We build layes of our cnn network as following using "mxnet" library.

- 1st convolutional layer
conv_1 = mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 =mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 = mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

- 2nd convolutional layer
conv_2 = mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 = mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 = mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

- 1st fully connected layer

flatten = mx.symbol.Flatten(data = pool_2)
fc_1=mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3= mx.symbol.Activation(data = fc_1, act_type = "tanh")

- 2nd fully connected layer
fc_2=mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)

