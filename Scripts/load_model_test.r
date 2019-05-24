library(keras)
library(tidyverse)
library(EBImage)
library(bioimagetools)

rdata_file <- "/scratch/network/ds65/ELE571/stress_outputs/ele571_3d_lowSNR_1.RData"
model_file <- "/scratch/network/ds65/ELE571/stress_outputs/ele571_3d_lowSNR_model_1.h5"

load(rdata_file)

model <- load_model_hdf5(model_file, 
                         custom_objects = c("dice_coef" = dice_coef, 
                                            "dice_coef_loss" = dice_coef_loss))

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

input_test <- sample_n(test_data[test_data$ImageId == "0001",], 1) %>%
            mutate(Y = map(MaskFile, preprocess_masks, new_shape = SHAPE),
                   X = map(ImageFile, preprocess_image, shape = SHAPE))

print(paste("Testing image ", input_test$ImageId[1],sep=""))

input_test <- input_test %>% select(X,Y)

W_test <- map(input_test$Y, function(x) {bw <- x > 0; 1*bw}) ##binarize
         
X_test <- list2tensor(input_test$X)
Y_test <- list2tensor(input_test$Y)
Ybw_test <- list2tensor(W_test)         

Y_hat_test_1 <- predict(model, x = X_test)

model %>% evaluate(X_test, Y_test)

save.image(file = "3d_lowSNR_stress_test0001_1.RData")
writeTIF(Y_hat_test_1[1,,,,], file = "test0001_prediction_lowSNR_1.tif")


