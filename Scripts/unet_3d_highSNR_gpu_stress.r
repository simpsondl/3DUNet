library(keras)
library(tidyverse)
library(EBImage)
library(bioimagetools)

options(EBImage.display = "raster")
use_condaenv("py36")

### Parameters
TRAIN_PATH <- "c00_highSNR_input/train/"
TEST_PATH <- "c00_highSNR_input/test/"
HEIGHT = 256
WIDTH = 256
STACKS = 128
CHANNELS = 1
SHAPE = c(WIDTH, HEIGHT, STACKS, CHANNELS)
BATCH_SIZE = 1
EPOCHS = 50

### Get training data
images_train <- list.files(paste(TRAIN_PATH,"image",sep = ""),
                           pattern = ".tif",
                           full.names = T)
masks_train <- list.files(paste(TRAIN_PATH,"mask",sep = ""),
                          pattern = ".tif",
                          full.names = T)

imageids <- gsub("^.*([0-9]{4}).*$","\\1",basename(images_train))

train_data <- data.frame("ImageId" = imageids,
                         "ImageFile" = images_train,
                         "MaskFile" = masks_train)
train_data <- train_data %>% 
  mutate(ImageShape = map(ImageFile, 
                          .f = function(file) dim(readImage(as.character(file)))))
train_data$ImageFile <- as.character(train_data$ImageFile)
train_data$MaskFile <- as.character(train_data$MaskFile)
train_data %>% glimpse()

## Preprocess original images
preprocess_image <- function(file, shape){
  image <- readTIF(file)
  image <- resize(image, w = shape[1], h = shape[2])  ## make all images of dimensions
  image <- (image - min(image))/(max(image) - min(image))         ## scale pixel values
  image <- clahe(image)                               ## local adaptive contrast enhancement
  image <- image[,,1:shape[3]]
  image <- array_reshape(image, SHAPE)
  imageData(image)                                    ## return as array
}

## Preprocess masks
preprocess_masks <- function(file, new_shape){
  require(bioimagetools)
  
  image <- readTIF(file)
  image <- resize(image, w = new_shape[1], h = new_shape[2])
  image <- (image - min(image))/(max(image) - min(image))
  image[image != 0] <- 1
  image <- image[,,1:new_shape[3]]
  image <- array_reshape(image, SHAPE)
  imageData(image)
}

####
# UNet implementation modified from:
# https://www.kaggle.com/mviterson/segmentation-using-r-keras-u-net-lb-0-294/notebook
####

## unet 2x2x2 3DConv layer
unet_layer <- function(object, filters, kernel_size = c(3, 3, 3),
                       padding = "same", kernel_initializer = "he_normal",
                       dropout = 0.1, activation="relu"){
  
  object %>%
    layer_conv_3d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
    layer_batch_normalization() %>%
    layer_activation(activation) %>%
    layer_spatial_dropout_3d(rate = dropout) %>%
    layer_conv_3d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
    layer_batch_normalization() %>%
    layer_activation(activation)
}

unet <- function(shape, nlevels = 4, nfilters = 16, dropouts = c(0.1, 0.1, 0.2, 0.2, 0.3)){
  
  message("Constructing U-Net with ", nlevels, " levels initial number of filters is: ", nfilters)
  
  filter_sizes <- nfilters*2^seq.int(0, nlevels)
  
  ## Loop over contracting layers
  clayers <- clayers_pooled <- list()
  
  ## inputs
  clayers_pooled[[1]] <- layer_input(shape = shape)
  
  for(i in 2:(nlevels+1)) {
    clayers[[i]] <- unet_layer(clayers_pooled[[i - 1]],
                               filters = filter_sizes[i - 1],
                               dropout = dropouts[i-1])
    
    clayers_pooled[[i]] <- layer_max_pooling_3d(clayers[[i]],
                                                pool_size = c(2, 2, 2),
                                                strides = c(2, 2, 2))
  }
  
  ## Loop over expanding layers
  elayers <- list()
  
  ## center
  elayers[[nlevels + 1]] <- unet_layer(clayers_pooled[[nlevels + 1]],
                                       filters = filter_sizes[nlevels + 1],
                                       dropout = dropouts[nlevels + 1])
  
  for(i in nlevels:1) {
    elayers[[i]] <- layer_conv_3d_transpose(elayers[[i+1]],
                                            filters = filter_sizes[i],
                                            kernel_size = c(2, 2, 2),
                                            strides = c(2, 2, 2),
                                            padding = "same")
    
    elayers[[i]] <- layer_concatenate(list(elayers[[i]], clayers[[i + 1]]), axis = 4)
    elayers[[i]] <- unet_layer(elayers[[i]], filters = filter_sizes[i], dropout = dropouts[i])
    
  }
  
  ## Output layer
  outputs <- layer_conv_3d(elayers[[1]], filters = 1, kernel_size = c(1, 1, 1), activation = "sigmoid")
  
  return(keras_model(inputs = clayers_pooled[[1]], outputs = outputs))
}

model <- unet(shape = SHAPE, 
              nlevels = 3, 
              nfilters = 16, 
              dropouts = c(0.1, 0.1, 0.2, 0.3))
summary(model)

dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
}
attr(dice_coef, "py_function_name") <- "dice_coef"

dice_coef_loss <- function(y_true, y_pred) -dice_coef(y_true, y_pred)
attr(dice_coef_loss, "py_function_name") <- "dice_coef_loss"

model %>% compile(optimizer = "adam", 
                  loss = dice_coef_loss, 
                  metrics = custom_metric("dice_coef",dice_coef))

set.seed(123)
input <- train_data %>%
  mutate(Y = map(MaskFile, preprocess_masks, new_shape = SHAPE),
         X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
  select(X,Y)

list2tensor <- function(xList) {
  xTensor <- simplify2array(xList)
  aperm(xTensor, c(5, 1, 2, 3, 4))    
}

W <- map(input$Y, function(x) {bw <- x > 0; 1*bw}) ##binarize

X <- list2tensor(input$X)
Y <- list2tensor(input$Y)
Ybw <- list2tensor(W)         
dim(Ybw)
dim(Y)
dim(X)

early_stopping <- callback_early_stopping(patience = 5)

history <- model %>%
  fit(X, Ybw,
      batch_size = BATCH_SIZE,
      epochs = EPOCHS,
      validation_split = 0.1)
#callbacks = list(early_stopping)

model %>% save_model_hdf5("/scratch/network/ds65/ELE571/stress_outputs/ele571_3d_highSNR_gpu_5_model.h5")
save.image(file = "/scratch/network/ds65/ELE571/stress_outputs/ele571_3d_highSNR_gpu_5.RData")

### Predict on training data for sanity
#print("Get prediction on a training sample...")
#Y_hat <- predict(model, x = reshape(X[1,,,,], c(1,256,256,128,1)))

### Predict on testing data
print("Get predictions on testing data...")
images_test <- list.files(paste(TEST_PATH,"image",sep = ""),
                         pattern = ".tif",
                         full.names = T)
masks_test <- list.files(paste(TEST_PATH,"mask",sep = ""),
                        pattern = ".tif",
                        full.names = T)

imageids <- gsub("^.*([0-9]{4}).*$","\\1",basename(images_test))

test_data <- data.frame("ImageId" = imageids,
                         "ImageFile" = images_test,
                         "MaskFile" = masks_test)
test_data <- test_data %>% 
                mutate(ImageShape = map(ImageFile, 
                                      .f = function(file) dim(readImage(as.character(file)))))
test_data$ImageFile <- as.character(test_data$ImageFile)
test_data$MaskFile <- as.character(test_data$MaskFile)
test_data %>% glimpse()

input_test <- sample_n(test_data, nrow(test_data)) %>%
            mutate(Y = map(MaskFile, preprocess_masks, new_shape = SHAPE),
                   X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
            select(X,Y)

W_test <- map(input_test$Y, function(x) {bw <- x > 0; 1*bw}) ##binarize
         
X_test <- list2tensor(input_test$X)
Y_test <- list2tensor(input_test$Y)
Ybw_test <- list2tensor(W_test)         

Y_hat_test <- predict(model, x = X_test)

### Get metrics
print("Metrics for test images...")
model %>% evaluate(X_test, Y_test)

