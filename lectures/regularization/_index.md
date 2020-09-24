---
title: Regularization in Deep Neural Networks
---

# Regularization in Deep Neural Networks


## Introduction 

![reg-strengths](images/reg_strengths_cs231n.jpeg)
   
In many cases we need to address overfitting by adding in the objective function some tunable penalty term that prevents it. Such penalty term is usually:

$\lambda J_{penalty} = \lambda \left(\sum_l W_{(l)}^2 \right) $

where $l$ is the hidden layer index and $W$ is the weight tensor. 

## Dropout

## Batch Normalization

## Combined Regularization strategies

https://arxiv.org/pdf/1801.05134.pdf
