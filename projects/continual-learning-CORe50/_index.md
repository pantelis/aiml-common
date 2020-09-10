---
title: Continual Learning 
---

# Continual Learning (CL) 

In this project you will implement continual learning and you are free to select a method from any of these three non-exclusive categories, [as described in detail](https://arxiv.org/abs/1802.07569). This is a [very active research area](https://sites.google.com/view/clvision2020/challenge?authuser=0). 

## Datasets and Tasks

![core50](images/core50.gif#center)

You are given two dataset options for this project as shown in the table below. The CORe50 option is more difficult than the MNIST option. Grading will happen relative to teams that selected the same option. The CORe50 option may require AWS GPU resources to run. MNIST is able to run in Colab with standard free accounts. If you have access to compute, you will learn more from selecting the CORe50 option as this is closest to a real life dataset than the rotated MNIST which is not even testing for new classes - rather it tests CL on rotated version of classes it has seen before.

|CORe50 Option   | Rotated MNIST Option  |
| --- | --- |
|  You will use [this](https://vlomonaco.github.io/core50/index.html) dataset and evaluate your method for New Class (NC) scenario.   |  Use the dataset provided [here](https://github.com/facebookresearch/GradientEpisodicMemory)  based on [this paper](http://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf) |
|   Object recognition (classification).  | Object recognition (classification). |

