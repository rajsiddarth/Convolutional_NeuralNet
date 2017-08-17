#
rm(list=ls(all=T))

#Getting Oivetti Faces data set
library(RCurl)

x=read.csv("https://raw.githubusercontent.com/rajsiddarth/
           Convolutional_NeuralNet/master/olivetti_X.csv",header=F)
y=read.csv("https://raw.githubusercontent.com/rajsiddarth/
                    Convolutional_NeuralNet/master/olivetti_y.csv",header=F)

library(EBImage)
final_Data=data.frame()
# For each image, resize and set it to greyscale
for(i in 1:nrow(x))
{
  # Try-catch
  result = tryCatch({
    # Image (as 1d vector)
    img = as.numeric(x[i,])
    # Reshape as a 64x64 image (EBImage object)
    img = Image(img, dim=c(64, 64), colormode = "Grayscale")
    # Resize image to 28x28 pixels
    img_resized = resize(img, w = 28, h = 28)
    # Get image matrix
    img_matrix = img_resized@.Data
    # Coerce to a vector
    img_vector = as.vector(t(img_matrix))
    # Add label
    label = y[i,]
    vec = c(label, img_vector)
    # Stack in rs_df using rbind
    final_Data = rbind(final_Data, vec)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function 
    error = function(e){print(e)})
}

# Set names. The first columns are the labels, the other columns are the pixels.
names(final_Data)=c("label", paste("pixel", c(1:784)))

# Splitting data into train test sets

# Set seed for reproducibility purposes
set.seed(1243)

shuffled =final_Data[sample(1:400),]
train=shuffled[1:360, ]
test=shuffled[361:400, ]

# Installing Mxnet package for R
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")

# Building the Model

# Set up train and test datasets
train= data.matrix(train)
train_x=t(train[, -1])
train_y=train[, 1]
train_array=train_x
dim(train_array)=c(28, 28, 1, ncol(train_x))


test_x= t(test[, -1])
test_y =test[, 1]
test_array =test_x
dim(test_array)= c(28, 28, 1, ncol(test_x))

# Set up the symbolic model
library(mxnet)
data=mx.symbol.Variable('data')

# 1st convolutional layer
conv_1 = mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 =mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 = mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 2nd convolutional layer
conv_2 = mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 = mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 = mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 1st fully connected layer

flatten = mx.symbol.Flatten(data = pool_2)
fc_1=mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3= mx.symbol.Activation(data = fc_1, act_type = "tanh")

# 2nd fully connected layer
fc_2=mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)

# Output. Softmax output since we'd like to get some probabilities.
model=mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
mx.set.seed(1234)

# Device used. CPU in my case.
devices =mx.cpu()

# Train the model
model=mx.model.FeedForward.create(model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 480,
                                     array.batch.size = 40,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Predicting on Test data
# Predict labels

predicted =predict(model, test_array)

# Assign labels

predicted_labels=max.col(t(predicted)) - 1

# Get accuracy
print(paste("accuracy =",(sum(diag(table(test[, 1], predicted_labels)))/40)*100,"%"))

