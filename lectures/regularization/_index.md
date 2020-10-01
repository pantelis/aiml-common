---
title: Regularization in Deep Neural Networks
---

# Regularization in Deep Neural Networks

In this chapter we look at the training aspects of DNNs and investigate schemes that can help us avoid overfitting a common trait of putting too much network capacity to the supervised learning problem at hand. We start from the realization that the Bayesian approach is the most elegant framework to study regularization. 

## Why the Bayesian framework can regularize

1. The strict formulation of Maximum Likelihood (ML) requires considering the complete dataset of size $m$.
2. Implementations approximate the ML solution via online algorithms such as the SGD and others. 
3. 

![reg-strengths](images/reg_strengths_cs231n.jpeg)
   
In many cases we need to address overfitting by adding in the objective function some tunable penalty term that prevents it. Such penalty term is usually:

$\lambda J_{penalty} = \lambda \left(\sum_l W_{(l)}^2 \right) $

where $l$ is the hidden layer index and $W$ is the weight tensor. 

## Dropout

## Batch Normalization

## Combined Regularization strategies

https://arxiv.org/pdf/1801.05134.pdf
