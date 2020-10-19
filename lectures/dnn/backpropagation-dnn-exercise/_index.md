---
title: Backpropagation DNN exercise
---

# Backpropagation DNN exercise

<img src="images/tensorflow.gif" width="60%">

_Computational graph in Tensorboard showing the components involved in a TF BP update_

A network consist of a concatenation of the following layers

1. Fully Connected layer with input $x^{(1)}$, $W^{(1)}$ and output $z^{(1)}$. 
2. RELU producing $a^{(1)}$
3. Fully Connected layer with parameters $W^{(2)}$ producing $z^{(2)}$
4. SOFTMAX producing $\hat{y}$
5. Cross-Entropy (CE) loss producing $L$

The task of backprop consists of the following steps:

1. Sketch the network and write down the equations for the forward path. 
2. Propagate the backwards path i.e. make sure you write down the expressions of the gradient of the loss with respect to all the network _parameters_. 

NOTE: Please note that we have omitted the bias terms for simplicity.