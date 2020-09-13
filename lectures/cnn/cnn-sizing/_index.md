---
title: CNN Sizing
weight: 65
draft: false
---

# CNN Sizing

Sizing is an exercise that will help you how to specify hyperparameters in ```tf.keras``` such as the height, width, depth of filters, feature map sizes etc. Sizing is needed so that you can stitch all the layers together correctly. 

## Sizing Example

We will use an toy network for such exercise. 

![convnet](images/convnet.jpeg)

The example CNN architecture above has the following layers:

* INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
* CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
* RELU layer will apply an elementwise activation function, such as the $\max(0,x)$ thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
* POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
* FC (i.e. fully-connected) layer, also known as dense, will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10 dataset. As you recall in FC layers each neuron in this layer will be connected to all the neurons in the previous volume.

The impact of padding on the sizing of the produced feature map is shown in the following numerical example. The example is for [28x28x3] input layer but results can be extrapolated for [32x32x3]

![sizing-example](images/sizing-example.png#center)

## Number of parameters and memory

![revolution-of-depth](images/revolution-of-depth.png#center)
