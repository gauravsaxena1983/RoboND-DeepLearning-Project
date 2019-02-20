
# Follow-Me Project
Congratulations on reaching the final project of the Robotics Nanodegree! 

Previously, you worked on the Semantic Segmentation lab where you built a deep learning network that locates a particular human target within an image. For this project, you will utilize what you implemented and learned from that lab and extend it to train a deep learning model that will allow a simulated quadcopter to follow around the person that it detects! 

Most of the code below is similar to the lab with some minor modifications. You can start with your existing solution, and modify and improve upon it to train the best possible model for this task.

You can click on any of the following to quickly jump to that part of this notebook:
1. [Data Collection](#data)
2. [FCN Layers](#fcn)
3. [Build the Model](#build)
4. [Training](#training)
5. [Prediction](#prediction)
6. [Evaluation](#evaluation)

## Data Collection<a id='data'></a>
We have provided you with a starting dataset for this project. Download instructions can be found in the README for this project's repo.
Alternatively, you can collect additional data of your own to improve your model. Check out the "Collecting Data" section in the Project Lesson in the Classroom for more details!


```python
import os
import glob
import sys
import tensorflow as tf

from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools
```

## FCN Layers <a id='fcn'></a>
In the Classroom, we discussed the different layers that constitute a fully convolutional network (FCN). The following code will introduce you to the functions that you need to build your semantic segmentation model.

### Separable Convolutions
The Encoder for your FCN will essentially require separable convolution layers, due to their advantages as explained in the classroom. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided below for your use. Each includes batch normalization with the ReLU activation function applied to the layers. 


```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

### Bilinear Upsampling
The following helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended, but you can try out different factors as well. Upsampling is used in the decoder block of the FCN.


```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

## Build the Model <a id='build'></a>
In the following cells, you will build an FCN to train a model to detect and locate the hero target within an image. The steps are:
- Create an `encoder_block`
- Create a `decoder_block`
- Build the FCN consisting of encoder block(s), a 1x1 convolution, and decoder block(s).  This step requires experimentation with different numbers of layers and filter sizes to build your model.

### Encoder Block
Create an encoder block that includes a separable convolution layer using the `separable_conv2d_batchnorm()` function. The `filters` parameter defines the size or depth of the output layer. For example, 32 or 64. 


```python
def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

### Decoder Block
The decoder block is comprised of three parts:
- A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
- A layer concatenation step. This step is similar to skip connections. You will concatenate the upsampled small_ip_layer and the large_ip_layer.
- Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.


```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled_small_ip_layer = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([upsampled_small_ip_layer, large_ip_layer])
    
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm( output_layer, filters, strides=1)
    
    return output_layer
```

### Model

Now that you have the encoder and decoder blocks ready, go ahead and build your FCN architecture! 

There are three steps:
- Add encoder blocks to build the encoder layers. This is similar to how you added regular convolutional layers in your CNN lab.
- Add a 1x1 Convolution layer using the conv2d_batchnorm() function. Remember that 1x1 Convolutions require a kernel and stride of 1.
- Add decoder blocks for the decoder layers.


```python
def fcn_model(inputs, num_classes):
    
    # Remember that with each encoder layer, the depth of your model (the number of filters) 
    encoder1 = encoder_block(inputs, filters=64, strides=2)
    print('layer1', encoder1)
    encoder2 = encoder_block(encoder1, filters=64, strides=2)
    print('layer2', encoder2)
    encoder3 = encoder_block(encoder2, filters=64, strides=2)
    print('layer3', encoder3)
    encoder4 = encoder_block(encoder3, filters=128, strides=2)
    print('layer4', encoder4)
    
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    onebyone_convolution_layer = conv2d_batchnorm(encoder4, filters=128, kernel_size=1, strides=1)
    print('layer5', onebyone_convolution_layer)                      
        
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder1 = decoder_block(onebyone_convolution_layer, encoder3, filters= 128)
    print('layer6', decoder1)
    decoder2 = decoder_block(decoder1, encoder2, filters=64)
    print('layer7', decoder2)
    decoder3 = decoder_block(decoder2, encoder1, filters=64)
    print('layer8', decoder3)
    decoder4 = decoder_block(decoder3, inputs, filters=64)
    print('layer9', decoder4)
    x = decoder4
    
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

## Training <a id='training'></a>
The following cells will use the FCN you created and define an ouput layer based on the size of the processed image and the number of classes recognized. You will define the hyperparameters to compile and train your model.

Please Note: For this project, the helper code in `data_iterator.py` will resize the copter images to 160x160x3 to speed up training.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)
```

    layer1 Tensor("batch_normalization/batchnorm/add_1:0", shape=(?, 80, 80, 64), dtype=float32)
    layer2 Tensor("batch_normalization_2/batchnorm/add_1:0", shape=(?, 40, 40, 64), dtype=float32)
    layer3 Tensor("batch_normalization_3/batchnorm/add_1:0", shape=(?, 20, 20, 64), dtype=float32)
    layer4 Tensor("batch_normalization_4/batchnorm/add_1:0", shape=(?, 10, 10, 128), dtype=float32)
    layer5 Tensor("batch_normalization_5/batchnorm/add_1:0", shape=(?, 10, 10, 128), dtype=float32)
    layer6 Tensor("batch_normalization_6/batchnorm/add_1:0", shape=(?, 20, 20, 128), dtype=float32)
    layer7 Tensor("batch_normalization_7/batchnorm/add_1:0", shape=(?, 40, 40, 64), dtype=float32)
    layer8 Tensor("batch_normalization_8/batchnorm/add_1:0", shape=(?, 80, 80, 64), dtype=float32)
    layer9 Tensor("batch_normalization_9/batchnorm/add_1:0", shape=(?, 160, 160, 64), dtype=float32)


### Hyperparameters
Define and tune your hyperparameters.
- **batch_size**: number of training samples/images that get propagated through the network in a single pass.
- **num_epochs**: number of times the entire training dataset gets propagated through the network.
- **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
- **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well.
- **workers**: maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with. 


```python
learning_rate = 0.001
batch_size = 100
num_epochs = 200
steps_per_epoch = 41
validation_steps = 12
workers = 4
```


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

from workspace_utils import active_session
# Keeping Your Session Active
with active_session():
    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                   data_folder=os.path.join('..', 'data', 'train'),
                                                   image_shape=image_shape,
                                                   shift_aug=True)

    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                 data_folder=os.path.join('..', 'data', 'validation'),
                                                 image_shape=image_shape)

    logger_cb = plotting_tools.LoggerPlotter()
    callbacks = [logger_cb]

    model.fit_generator(train_iter,
                        steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                        epochs = num_epochs, # the number of epochs to train for,
                        validation_data = val_iter, # validation iterator
                        validation_steps = validation_steps, # the number of batches to validate on
                        callbacks=callbacks,
                        workers = workers)
```

    Epoch 1/200
    40/41 [============================>.] - ETA: 1s - loss: 0.9818


![png](/misc_img/output_19.png)


    41/41 [==============================] - 62s - loss: 0.9761 - val_loss: 0.7409
    Epoch 2/200
    40/41 [============================>.] - ETA: 1s - loss: 0.5427


![png](output_19_3.png)


    41/41 [==============================] - 58s - loss: 0.5380 - val_loss: 0.3705
    Epoch 3/200
    40/41 [============================>.] - ETA: 1s - loss: 0.2387


![png](output_19_5.png)


    41/41 [==============================] - 57s - loss: 0.2366 - val_loss: 0.1898
    Epoch 4/200
    40/41 [============================>.] - ETA: 1s - loss: 0.1181


![png](output_19_7.png)


    41/41 [==============================] - 57s - loss: 0.1174 - val_loss: 0.1336
    Epoch 5/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0803


![png](output_19_9.png)


    41/41 [==============================] - 58s - loss: 0.0802 - val_loss: 0.1200
    Epoch 6/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0627


![png](/misc_img/output_191.png)


    41/41 [==============================] - 58s - loss: 0.0627 - val_loss: 0.1072
    Epoch 7/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0532


![png](/misc_img/output_193.png)


    41/41 [==============================] - 57s - loss: 0.0533 - val_loss: 0.1056
    Epoch 8/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0469


![png](/misc_img/output_195.png)


    41/41 [==============================] - 57s - loss: 0.0467 - val_loss: 0.1010
    Epoch 9/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0521


![png](/misc_img/output_197.png)


    41/41 [==============================] - 58s - loss: 0.0519 - val_loss: 0.1023
    Epoch 10/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0414


![png](/misc_img/output_199.png)


    41/41 [==============================] - 57s - loss: 0.0412 - val_loss: 0.1011
    Epoch 11/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0372


![png](output_19_21.png)


    41/41 [==============================] - 57s - loss: 0.0370 - val_loss: 0.1001
    Epoch 12/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0351


![png](output_19_23.png)


    41/41 [==============================] - 57s - loss: 0.0350 - val_loss: 0.0958
    Epoch 13/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0327


![png](output_19_25.png)


    41/41 [==============================] - 57s - loss: 0.0328 - val_loss: 0.0841
    Epoch 14/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0310


![png](output_19_27.png)


    41/41 [==============================] - 57s - loss: 0.0310 - val_loss: 0.0766
    Epoch 15/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0294


![png](output_19_29.png)


    41/41 [==============================] - 57s - loss: 0.0295 - val_loss: 0.0560
    Epoch 16/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0293


![png](output_19_31.png)


    41/41 [==============================] - 58s - loss: 0.0294 - val_loss: 0.0520
    Epoch 17/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0281


![png](output_19_33.png)


    41/41 [==============================] - 57s - loss: 0.0280 - val_loss: 0.0378
    Epoch 18/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0266


![png](output_19_35.png)


    41/41 [==============================] - 57s - loss: 0.0266 - val_loss: 0.0396
    Epoch 19/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0260


![png](output_19_37.png)


    41/41 [==============================] - 57s - loss: 0.0259 - val_loss: 0.0364
    Epoch 20/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0243


![png](output_19_39.png)


    41/41 [==============================] - 58s - loss: 0.0243 - val_loss: 0.0498
    Epoch 21/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0249


![png](output_19_41.png)


    41/41 [==============================] - 58s - loss: 0.0250 - val_loss: 0.0310
    Epoch 22/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0234


![png](output_19_43.png)


    41/41 [==============================] - 58s - loss: 0.0234 - val_loss: 0.0359
    Epoch 23/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0217


![png](output_19_45.png)


    41/41 [==============================] - 57s - loss: 0.0217 - val_loss: 0.0326
    Epoch 24/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0235


![png](output_19_47.png)


    41/41 [==============================] - 57s - loss: 0.0234 - val_loss: 0.0337
    Epoch 25/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0211


![png](output_19_49.png)


    41/41 [==============================] - 57s - loss: 0.0210 - val_loss: 0.0315
    Epoch 26/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0224


![png](output_19_51.png)


    41/41 [==============================] - 58s - loss: 0.0224 - val_loss: 0.0367
    Epoch 27/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0213


![png](output_19_53.png)


    41/41 [==============================] - 57s - loss: 0.0213 - val_loss: 0.0297
    Epoch 28/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0195


![png](output_19_55.png)


    41/41 [==============================] - 57s - loss: 0.0194 - val_loss: 0.0307
    Epoch 29/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0200


![png](output_19_57.png)


    41/41 [==============================] - 57s - loss: 0.0200 - val_loss: 0.0246
    Epoch 30/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0204


![png](output_19_59.png)


    41/41 [==============================] - 57s - loss: 0.0204 - val_loss: 0.0356
    Epoch 31/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0211


![png](output_19_61.png)


    41/41 [==============================] - 57s - loss: 0.0211 - val_loss: 0.0335
    Epoch 32/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0199


![png](output_19_63.png)


    41/41 [==============================] - 57s - loss: 0.0198 - val_loss: 0.0295
    Epoch 33/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0187


![png](output_19_65.png)


    41/41 [==============================] - 57s - loss: 0.0187 - val_loss: 0.0393
    Epoch 34/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0185


![png](output_19_67.png)


    41/41 [==============================] - 58s - loss: 0.0186 - val_loss: 0.0294
    Epoch 35/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0179


![png](output_19_69.png)


    41/41 [==============================] - 57s - loss: 0.0179 - val_loss: 0.0317
    Epoch 36/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0175


![png](output_19_71.png)


    41/41 [==============================] - 57s - loss: 0.0175 - val_loss: 0.0370
    Epoch 37/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0176


![png](output_19_73.png)


    41/41 [==============================] - 57s - loss: 0.0176 - val_loss: 0.0310
    Epoch 38/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0189


![png](output_19_75.png)


    41/41 [==============================] - 57s - loss: 0.0190 - val_loss: 0.0363
    Epoch 39/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0188


![png](output_19_77.png)


    41/41 [==============================] - 57s - loss: 0.0187 - val_loss: 0.0318
    Epoch 40/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0180


![png](output_19_79.png)


    41/41 [==============================] - 57s - loss: 0.0180 - val_loss: 0.0296
    Epoch 41/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0167


![png](output_19_81.png)


    41/41 [==============================] - 57s - loss: 0.0167 - val_loss: 0.0310
    Epoch 42/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0167


![png](output_19_83.png)


    41/41 [==============================] - 57s - loss: 0.0167 - val_loss: 0.0313
    Epoch 43/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0169


![png](output_19_85.png)


    41/41 [==============================] - 58s - loss: 0.0169 - val_loss: 0.0293
    Epoch 44/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0159


![png](output_19_87.png)


    41/41 [==============================] - 59s - loss: 0.0159 - val_loss: 0.0310
    Epoch 45/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0164


![png](output_19_89.png)


    41/41 [==============================] - 57s - loss: 0.0163 - val_loss: 0.0272
    Epoch 46/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0156


![png](output_19_91.png)


    41/41 [==============================] - 58s - loss: 0.0156 - val_loss: 0.0299
    Epoch 47/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0155


![png](output_19_93.png)


    41/41 [==============================] - 57s - loss: 0.0156 - val_loss: 0.0297
    Epoch 48/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0151


![png](output_19_95.png)


    41/41 [==============================] - 58s - loss: 0.0151 - val_loss: 0.0298
    Epoch 49/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0181


![png](output_19_97.png)


    41/41 [==============================] - 58s - loss: 0.0181 - val_loss: 0.0306
    Epoch 50/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0168


![png](output_19_99.png)


    41/41 [==============================] - 58s - loss: 0.0167 - val_loss: 0.0324
    Epoch 51/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0157


![png](/misc_img/output_1901.png)


    41/41 [==============================] - 57s - loss: 0.0156 - val_loss: 0.0371
    Epoch 52/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0161


![png](/misc_img/output_1903.png)


    41/41 [==============================] - 58s - loss: 0.0160 - val_loss: 0.0309
    Epoch 53/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0150


![png](/misc_img/output_1905.png)


    41/41 [==============================] - 58s - loss: 0.0150 - val_loss: 0.0314
    Epoch 54/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0145


![png](/misc_img/output_1907.png)


    41/41 [==============================] - 58s - loss: 0.0145 - val_loss: 0.0300
    Epoch 55/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0142


![png](/misc_img/output_1909.png)


    41/41 [==============================] - 57s - loss: 0.0143 - val_loss: 0.0292
    Epoch 56/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0144


![png](/misc_img/output_1911.png)


    41/41 [==============================] - 57s - loss: 0.0144 - val_loss: 0.0284
    Epoch 57/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0158


![png](/misc_img/output_1913.png)


    41/41 [==============================] - 57s - loss: 0.0158 - val_loss: 0.0266
    Epoch 58/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0144


![png](/misc_img/output_1915.png)


    41/41 [==============================] - 57s - loss: 0.0144 - val_loss: 0.0285
    Epoch 59/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0141


![png](/misc_img/output_1917.png)


    41/41 [==============================] - 57s - loss: 0.0141 - val_loss: 0.0262
    Epoch 60/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0135


![png](/misc_img/output_1919.png)


    41/41 [==============================] - 57s - loss: 0.0135 - val_loss: 0.0337
    Epoch 61/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0163


![png](/misc_img/output_1921.png)


    41/41 [==============================] - 57s - loss: 0.0165 - val_loss: 0.0376
    Epoch 62/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0181


![png](/misc_img/output_1923.png)


    41/41 [==============================] - 57s - loss: 0.0180 - val_loss: 0.0323
    Epoch 63/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0148


![png](/misc_img/output_1925.png)


    41/41 [==============================] - 57s - loss: 0.0148 - val_loss: 0.0226
    Epoch 64/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0146


![png](/misc_img/output_1927.png)


    41/41 [==============================] - 57s - loss: 0.0145 - val_loss: 0.0295
    Epoch 65/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0136


![png](/misc_img/output_1929.png)


    41/41 [==============================] - 57s - loss: 0.0136 - val_loss: 0.0378
    Epoch 66/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0135


![png](/misc_img/output_1931.png)


    41/41 [==============================] - 57s - loss: 0.0134 - val_loss: 0.0230
    Epoch 67/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0134


![png](/misc_img/output_1933.png)


    41/41 [==============================] - 57s - loss: 0.0134 - val_loss: 0.0289
    Epoch 68/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0135


![png](/misc_img/output_1935.png)


    41/41 [==============================] - 57s - loss: 0.0135 - val_loss: 0.0372
    Epoch 69/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0132


![png](/misc_img/output_1937.png)


    41/41 [==============================] - 57s - loss: 0.0132 - val_loss: 0.0300
    Epoch 70/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0132


![png](/misc_img/output_1939.png)


    41/41 [==============================] - 58s - loss: 0.0132 - val_loss: 0.0298
    Epoch 71/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0129


![png](/misc_img/output_1941.png)


    41/41 [==============================] - 57s - loss: 0.0129 - val_loss: 0.0395
    Epoch 72/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0128


![png](/misc_img/output_1943.png)


    41/41 [==============================] - 58s - loss: 0.0127 - val_loss: 0.0306
    Epoch 73/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0127


![png](/misc_img/output_1945.png)


    41/41 [==============================] - 58s - loss: 0.0127 - val_loss: 0.0304
    Epoch 74/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0129


![png](/misc_img/output_1947.png)


    41/41 [==============================] - 57s - loss: 0.0129 - val_loss: 0.0330
    Epoch 75/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0138


![png](/misc_img/output_1949.png)


    41/41 [==============================] - 58s - loss: 0.0137 - val_loss: 0.0267
    Epoch 76/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0131


![png](/misc_img/output_1951.png)


    41/41 [==============================] - 57s - loss: 0.0131 - val_loss: 0.0302
    Epoch 77/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0128


![png](/misc_img/output_1953.png)


    41/41 [==============================] - 57s - loss: 0.0128 - val_loss: 0.0291
    Epoch 78/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0123


![png](/misc_img/output_1955.png)


    41/41 [==============================] - 57s - loss: 0.0123 - val_loss: 0.0203
    Epoch 79/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0129


![png](/misc_img/output_1957.png)


    41/41 [==============================] - 57s - loss: 0.0129 - val_loss: 0.0298
    Epoch 80/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0129


![png](/misc_img/output_1959.png)


    41/41 [==============================] - 58s - loss: 0.0129 - val_loss: 0.0313
    Epoch 81/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0125


![png](/misc_img/output_1961.png)


    41/41 [==============================] - 57s - loss: 0.0125 - val_loss: 0.0370
    Epoch 82/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0122


![png](/misc_img/output_1963.png)


    41/41 [==============================] - 57s - loss: 0.0122 - val_loss: 0.0307
    Epoch 83/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0122


![png](/misc_img/output_1965.png)


    41/41 [==============================] - 58s - loss: 0.0122 - val_loss: 0.0300
    Epoch 84/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0121


![png](/misc_img/output_1967.png)


    41/41 [==============================] - 57s - loss: 0.0121 - val_loss: 0.0324
    Epoch 85/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0120


![png](/misc_img/output_1969.png)


    41/41 [==============================] - 57s - loss: 0.0120 - val_loss: 0.0308
    Epoch 86/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0119


![png](/misc_img/output_1971.png)


    41/41 [==============================] - 57s - loss: 0.0119 - val_loss: 0.0331
    Epoch 87/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0118


![png](/misc_img/output_1973.png)


    41/41 [==============================] - 58s - loss: 0.0118 - val_loss: 0.0336
    Epoch 88/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0117


![png](/misc_img/output_1975.png)


    41/41 [==============================] - 58s - loss: 0.0117 - val_loss: 0.0301
    Epoch 89/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0115


![png](/misc_img/output_1977.png)


    41/41 [==============================] - 57s - loss: 0.0115 - val_loss: 0.0224
    Epoch 90/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0115


![png](/misc_img/output_1979.png)


    41/41 [==============================] - 57s - loss: 0.0114 - val_loss: 0.0391
    Epoch 91/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0116


![png](/misc_img/output_1981.png)


    41/41 [==============================] - 57s - loss: 0.0116 - val_loss: 0.0315
    Epoch 92/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0114


![png](/misc_img/output_1983.png)


    41/41 [==============================] - 57s - loss: 0.0114 - val_loss: 0.0320
    Epoch 93/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0116


![png](/misc_img/output_1985.png)


    41/41 [==============================] - 57s - loss: 0.0116 - val_loss: 0.0417
    Epoch 94/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0114


![png](/misc_img/output_1987.png)


    41/41 [==============================] - 57s - loss: 0.0115 - val_loss: 0.0312
    Epoch 95/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0112


![png](/misc_img/output_1989.png)


    41/41 [==============================] - 57s - loss: 0.0112 - val_loss: 0.0309
    Epoch 96/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0114


![png](/misc_img/output_1991.png)


    41/41 [==============================] - 57s - loss: 0.0114 - val_loss: 0.0398
    Epoch 97/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0117


![png](/misc_img/output_1993.png)


    41/41 [==============================] - 57s - loss: 0.0117 - val_loss: 0.0326
    Epoch 98/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0116


![png](/misc_img/output_1995.png)


    41/41 [==============================] - 57s - loss: 0.0116 - val_loss: 0.0366
    Epoch 99/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0113


![png](/misc_img/output_1997.png)


    41/41 [==============================] - 57s - loss: 0.0113 - val_loss: 0.0377
    Epoch 100/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0112


![png](/misc_img/output_1999.png)


    41/41 [==============================] - 57s - loss: 0.0112 - val_loss: 0.0326
    Epoch 101/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0111


![png](output_19_201.png)


    41/41 [==============================] - 58s - loss: 0.0111 - val_loss: 0.0414
    Epoch 102/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0111


![png](output_19_203.png)


    41/41 [==============================] - 57s - loss: 0.0111 - val_loss: 0.0301
    Epoch 103/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0112


![png](output_19_205.png)


    41/41 [==============================] - 57s - loss: 0.0112 - val_loss: 0.0316
    Epoch 104/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0115


![png](output_19_207.png)


    41/41 [==============================] - 57s - loss: 0.0115 - val_loss: 0.0361
    Epoch 105/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0263


![png](output_19_209.png)


    41/41 [==============================] - 57s - loss: 0.0264 - val_loss: 0.4565
    Epoch 106/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0275


![png](output_19_211.png)


    41/41 [==============================] - 57s - loss: 0.0274 - val_loss: 0.0599
    Epoch 107/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0228


![png](output_19_213.png)


    41/41 [==============================] - 57s - loss: 0.0227 - val_loss: 0.0425
    Epoch 108/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0196


![png](output_19_215.png)


    41/41 [==============================] - 58s - loss: 0.0196 - val_loss: 0.0357
    Epoch 109/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0169


![png](output_19_217.png)


    41/41 [==============================] - 57s - loss: 0.0169 - val_loss: 0.0297
    Epoch 110/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0169


![png](output_19_219.png)


    41/41 [==============================] - 57s - loss: 0.0169 - val_loss: 0.0260
    Epoch 111/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0154


![png](output_19_221.png)


    41/41 [==============================] - 57s - loss: 0.0153 - val_loss: 0.0297
    Epoch 112/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0142


![png](output_19_223.png)


    41/41 [==============================] - 57s - loss: 0.0142 - val_loss: 0.0292
    Epoch 113/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0156


![png](output_19_225.png)


    41/41 [==============================] - 57s - loss: 0.0156 - val_loss: 0.0338
    Epoch 114/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0156


![png](output_19_227.png)


    41/41 [==============================] - 57s - loss: 0.0155 - val_loss: 0.0277
    Epoch 115/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0145


![png](output_19_229.png)


    41/41 [==============================] - 57s - loss: 0.0145 - val_loss: 0.0290
    Epoch 116/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0138


![png](output_19_231.png)


    41/41 [==============================] - 56s - loss: 0.0137 - val_loss: 0.0287
    Epoch 117/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0130


![png](output_19_233.png)


    41/41 [==============================] - 57s - loss: 0.0130 - val_loss: 0.0216
    Epoch 118/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0125


![png](output_19_235.png)


    41/41 [==============================] - 57s - loss: 0.0125 - val_loss: 0.0278
    Epoch 119/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0125


![png](output_19_237.png)


    41/41 [==============================] - 58s - loss: 0.0125 - val_loss: 0.0309
    Epoch 120/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0124


![png](output_19_239.png)


    41/41 [==============================] - 58s - loss: 0.0125 - val_loss: 0.0253
    Epoch 121/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0123


![png](output_19_241.png)


    41/41 [==============================] - 57s - loss: 0.0123 - val_loss: 0.0286
    Epoch 122/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0119


![png](output_19_243.png)


    41/41 [==============================] - 58s - loss: 0.0118 - val_loss: 0.0310
    Epoch 123/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0123


![png](output_19_245.png)


    41/41 [==============================] - 58s - loss: 0.0123 - val_loss: 0.0294
    Epoch 124/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0119


![png](output_19_247.png)


    41/41 [==============================] - 57s - loss: 0.0119 - val_loss: 0.0299
    Epoch 125/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0116


![png](output_19_249.png)


    41/41 [==============================] - 58s - loss: 0.0117 - val_loss: 0.0252
    Epoch 126/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0115


![png](output_19_251.png)


    41/41 [==============================] - 57s - loss: 0.0115 - val_loss: 0.0191
    Epoch 127/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0114


![png](output_19_253.png)


    41/41 [==============================] - 57s - loss: 0.0114 - val_loss: 0.0309
    Epoch 128/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0113


![png](output_19_255.png)


    41/41 [==============================] - 58s - loss: 0.0113 - val_loss: 0.0296
    Epoch 129/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0141


![png](output_19_257.png)


    41/41 [==============================] - 58s - loss: 0.0141 - val_loss: 0.0306
    Epoch 130/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0172


![png](output_19_259.png)


    41/41 [==============================] - 58s - loss: 0.0175 - val_loss: 0.0323
    Epoch 131/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0147


![png](output_19_261.png)


    41/41 [==============================] - 57s - loss: 0.0147 - val_loss: 0.0316
    Epoch 132/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0124


![png](output_19_263.png)


    41/41 [==============================] - 57s - loss: 0.0124 - val_loss: 0.0207
    Epoch 133/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0123


![png](output_19_265.png)


    41/41 [==============================] - 58s - loss: 0.0123 - val_loss: 0.0294
    Epoch 134/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0117


![png](output_19_267.png)


    41/41 [==============================] - 58s - loss: 0.0117 - val_loss: 0.0292
    Epoch 135/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0115


![png](output_19_269.png)


    41/41 [==============================] - 57s - loss: 0.0115 - val_loss: 0.0290
    Epoch 136/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0114


![png](output_19_271.png)


    41/41 [==============================] - 58s - loss: 0.0114 - val_loss: 0.0284
    Epoch 137/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0111


![png](output_19_273.png)


    41/41 [==============================] - 58s - loss: 0.0110 - val_loss: 0.0342
    Epoch 138/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0117


![png](output_19_275.png)


    41/41 [==============================] - 58s - loss: 0.0117 - val_loss: 0.0230
    Epoch 139/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0112


![png](output_19_277.png)


    41/41 [==============================] - 58s - loss: 0.0112 - val_loss: 0.0298
    Epoch 140/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0111


![png](output_19_279.png)


    41/41 [==============================] - 58s - loss: 0.0111 - val_loss: 0.0303
    Epoch 141/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0107


![png](output_19_281.png)


    41/41 [==============================] - 57s - loss: 0.0107 - val_loss: 0.0352
    Epoch 142/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0108


![png](output_19_283.png)


    41/41 [==============================] - 57s - loss: 0.0108 - val_loss: 0.0303
    Epoch 143/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0107


![png](output_19_285.png)


    41/41 [==============================] - 57s - loss: 0.0107 - val_loss: 0.0313
    Epoch 144/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0107


![png](output_19_287.png)


    41/41 [==============================] - 58s - loss: 0.0107 - val_loss: 0.0340
    Epoch 145/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0108


![png](output_19_289.png)


    41/41 [==============================] - 58s - loss: 0.0108 - val_loss: 0.0311
    Epoch 146/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0109


![png](output_19_291.png)


    41/41 [==============================] - 58s - loss: 0.0109 - val_loss: 0.0250
    Epoch 147/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0105


![png](output_19_293.png)


    41/41 [==============================] - 57s - loss: 0.0105 - val_loss: 0.0354
    Epoch 148/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0107


![png](output_19_295.png)


    41/41 [==============================] - 57s - loss: 0.0107 - val_loss: 0.0317
    Epoch 149/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0105


![png](output_19_297.png)


    41/41 [==============================] - 58s - loss: 0.0105 - val_loss: 0.0272
    Epoch 150/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0108


![png](output_19_299.png)


    41/41 [==============================] - 57s - loss: 0.0108 - val_loss: 0.0209
    Epoch 151/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0109


![png](output_19_301.png)


    41/41 [==============================] - 57s - loss: 0.0109 - val_loss: 0.0314
    Epoch 152/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0103


![png](output_19_303.png)


    41/41 [==============================] - 57s - loss: 0.0103 - val_loss: 0.0330
    Epoch 153/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0105


![png](output_19_305.png)


    41/41 [==============================] - 57s - loss: 0.0105 - val_loss: 0.0410
    Epoch 154/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0103


![png](output_19_307.png)


    41/41 [==============================] - 58s - loss: 0.0103 - val_loss: 0.0320
    Epoch 155/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0102


![png](output_19_309.png)


    41/41 [==============================] - 57s - loss: 0.0102 - val_loss: 0.0311
    Epoch 156/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0103


![png](output_19_311.png)


    41/41 [==============================] - 57s - loss: 0.0103 - val_loss: 0.0412
    Epoch 157/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0107


![png](output_19_313.png)


    41/41 [==============================] - 58s - loss: 0.0106 - val_loss: 0.0344
    Epoch 158/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0133


![png](output_19_315.png)


    41/41 [==============================] - 58s - loss: 0.0133 - val_loss: 0.0395
    Epoch 159/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0159


![png](output_19_317.png)


    41/41 [==============================] - 58s - loss: 0.0159 - val_loss: 0.0333
    Epoch 160/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0131


![png](output_19_319.png)


    41/41 [==============================] - 58s - loss: 0.0131 - val_loss: 0.0309
    Epoch 161/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0114


![png](output_19_321.png)


    41/41 [==============================] - 57s - loss: 0.0114 - val_loss: 0.0200
    Epoch 162/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0108


![png](output_19_323.png)


    41/41 [==============================] - 57s - loss: 0.0108 - val_loss: 0.0409
    Epoch 163/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0106


![png](output_19_325.png)


    41/41 [==============================] - 58s - loss: 0.0106 - val_loss: 0.0300
    Epoch 164/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0105


![png](output_19_327.png)


    41/41 [==============================] - 58s - loss: 0.0105 - val_loss: 0.0402
    Epoch 165/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0103


![png](output_19_329.png)


    41/41 [==============================] - 58s - loss: 0.0103 - val_loss: 0.0280
    Epoch 166/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0103


![png](output_19_331.png)


    41/41 [==============================] - 58s - loss: 0.0104 - val_loss: 0.0313
    Epoch 167/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0101


![png](output_19_333.png)


    41/41 [==============================] - 58s - loss: 0.0101 - val_loss: 0.0224
    Epoch 168/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0101


![png](output_19_335.png)


    41/41 [==============================] - 58s - loss: 0.0101 - val_loss: 0.0336
    Epoch 169/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0100


![png](output_19_337.png)


    41/41 [==============================] - 59s - loss: 0.0101 - val_loss: 0.0313
    Epoch 170/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0100


![png](output_19_339.png)


    41/41 [==============================] - 59s - loss: 0.0101 - val_loss: 0.0400
    Epoch 171/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0100


![png](output_19_341.png)


    41/41 [==============================] - 58s - loss: 0.0100 - val_loss: 0.0282
    Epoch 172/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0100


![png](output_19_343.png)


    41/41 [==============================] - 57s - loss: 0.0100 - val_loss: 0.0316
    Epoch 173/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0098


![png](output_19_345.png)


    41/41 [==============================] - 58s - loss: 0.0098 - val_loss: 0.0441
    Epoch 174/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0101


![png](output_19_347.png)


    41/41 [==============================] - 58s - loss: 0.0102 - val_loss: 0.0238
    Epoch 175/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0099


![png](output_19_349.png)


    41/41 [==============================] - 57s - loss: 0.0099 - val_loss: 0.0328
    Epoch 176/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0105


![png](output_19_351.png)


    41/41 [==============================] - 57s - loss: 0.0104 - val_loss: 0.0345
    Epoch 177/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0107


![png](output_19_353.png)


    41/41 [==============================] - 58s - loss: 0.0107 - val_loss: 0.0406
    Epoch 178/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0121


![png](output_19_355.png)


    41/41 [==============================] - 57s - loss: 0.0122 - val_loss: 0.0434
    Epoch 179/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0142


![png](output_19_357.png)


    41/41 [==============================] - 57s - loss: 0.0142 - val_loss: 0.0270
    Epoch 180/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0115


![png](output_19_359.png)


    41/41 [==============================] - 57s - loss: 0.0115 - val_loss: 0.0233
    Epoch 181/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0111


![png](output_19_361.png)


    41/41 [==============================] - 57s - loss: 0.0111 - val_loss: 0.0309
    Epoch 182/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0104


![png](output_19_363.png)


    41/41 [==============================] - 57s - loss: 0.0104 - val_loss: 0.0377
    Epoch 183/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0101


![png](output_19_365.png)


    41/41 [==============================] - 58s - loss: 0.0100 - val_loss: 0.0268
    Epoch 184/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0099


![png](output_19_367.png)


    41/41 [==============================] - 57s - loss: 0.0099 - val_loss: 0.0312
    Epoch 185/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0097


![png](output_19_369.png)


    41/41 [==============================] - 58s - loss: 0.0098 - val_loss: 0.0200
    Epoch 186/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0099


![png](output_19_371.png)


    41/41 [==============================] - 57s - loss: 0.0099 - val_loss: 0.0227
    Epoch 187/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0097


![png](output_19_373.png)


    41/41 [==============================] - 57s - loss: 0.0098 - val_loss: 0.0317
    Epoch 188/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0097


![png](output_19_375.png)


    41/41 [==============================] - 58s - loss: 0.0097 - val_loss: 0.0313
    Epoch 189/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0096


![png](output_19_377.png)


    41/41 [==============================] - 57s - loss: 0.0096 - val_loss: 0.0305
    Epoch 190/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0098


![png](output_19_379.png)


    41/41 [==============================] - 57s - loss: 0.0098 - val_loss: 0.0322
    Epoch 191/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0095


![png](output_19_381.png)


    41/41 [==============================] - 57s - loss: 0.0096 - val_loss: 0.0317
    Epoch 192/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0096


![png](output_19_383.png)


    41/41 [==============================] - 57s - loss: 0.0096 - val_loss: 0.0351
    Epoch 193/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0096


![png](output_19_385.png)


    41/41 [==============================] - 57s - loss: 0.0096 - val_loss: 0.0337
    Epoch 194/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0096


![png](output_19_387.png)


    41/41 [==============================] - 57s - loss: 0.0096 - val_loss: 0.0379
    Epoch 195/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0095


![png](output_19_389.png)


    41/41 [==============================] - 58s - loss: 0.0095 - val_loss: 0.0338
    Epoch 196/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0095


![png](output_19_391.png)


    41/41 [==============================] - 58s - loss: 0.0096 - val_loss: 0.0341
    Epoch 197/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0096


![png](output_19_393.png)


    41/41 [==============================] - 58s - loss: 0.0096 - val_loss: 0.0453
    Epoch 198/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0095


![png](output_19_395.png)


    41/41 [==============================] - 57s - loss: 0.0095 - val_loss: 0.0326
    Epoch 199/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0095


![png](output_19_397.png)


    41/41 [==============================] - 58s - loss: 0.0095 - val_loss: 0.0335
    Epoch 200/200
    40/41 [============================>.] - ETA: 1s - loss: 0.0093


![png](output_19_399.png)


    41/41 [==============================] - 57s - loss: 0.0093 - val_loss: 0.0391



```python
# Save your trained model weights
weight_file_name = 'model_weights'
model_tools.save_network(model, weight_file_name)
```

## Prediction <a id='prediction'></a>

Now that you have your model trained and saved, you can make predictions on your validation dataset. These predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well your model is doing under different conditions.

There are three different predictions available from the helper code provided:
- **patrol_with_targ**: Test how well the network can detect the hero from a distance.
- **patrol_non_targ**: Test how often the network makes a mistake and identifies the wrong person as the target.
- **following_images**: Test how well the network can identify the target while following them.


```python
# If you need to load a model which you previously trained you can uncomment the codeline that calls the function below.

# weight_file_name = 'model_weights'
# restored_model = model_tools.load_network(weight_file_name)
```

The following cell will write predictions to files and return paths to the appropriate directories.
The `run_num` parameter is used to define or group all the data for a particular model run. You can change it for different runs. For example, 'run_1', 'run_2' etc.


```python
run_num = 'run_1'

val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data') 

val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                        run_num,'patrol_non_targ', 'sample_evaluation_data') 

val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')
```

Now lets look at your predictions, and compare them to the ground truth labels and original images.
Run each of the following cells to visualize some sample images from the predictions in the validation set.


```python
# images while following the target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
    
```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)



```python
# images while at patrol without target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
 
```


![png](output_27_0.png)



![png](output_27_1.png)



![png](output_27_2.png)



```python
   
# images while at patrol with target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
```


![png](output_28_0.png)



![png](output_28_1.png)



![png](output_28_2.png)


## Evaluation <a id='evaluation'></a>
Evaluate your model! The following cells include several different scores to help you evaluate your model under the different conditions discussed during the Prediction step. 


```python
# Scores for while the quad is following behind the target. 
true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
```

    number of validation samples intersection over the union evaulated on 542
    average intersection over union for background is 0.9956843761304677
    average intersection over union for other people is 0.3562358465661956
    average intersection over union for the hero is 0.9146969787739379
    number true positives: 539, number false positives: 0, number false negatives: 0



```python
# Scores for images while the quad is on patrol and the target is not visable
true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
```

    number of validation samples intersection over the union evaulated on 270
    average intersection over union for background is 0.9865713867317022
    average intersection over union for other people is 0.7310113795759405
    average intersection over union for the hero is 0.0
    number true positives: 0, number false positives: 37, number false negatives: 0



```python
# This score measures how well the neural network can detect the target from far away
true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)
```

    number of validation samples intersection over the union evaulated on 322
    average intersection over union for background is 0.9961992915941345
    average intersection over union for other people is 0.4319008679308681
    average intersection over union for the hero is 0.14966698909621085
    number true positives: 103, number false positives: 2, number false negatives: 198



```python
# Sum all the true positives, etc from the three datasets to get a weight for the score
true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
print(weight)
```

    0.7303754266211604



```python
# The IoU for the dataset that never includes the hero is excluded from grading
final_IoU = (iou1 + iou3)/2
print(final_IoU)
```

    0.532181983935



```python
# And the final grade score is 
final_score = final_IoU * weight
print(final_score)
```

    0.388692643557

