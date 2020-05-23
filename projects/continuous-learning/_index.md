---
title: Continual Learning for Robotic Perception
---

# Continual Learning (CL) for Robotic Perception

**This project is due March 29 at 11:59pm**

## Introduction 

One of the greatest goals of AI is building an artificial continual learning agent which can construct a sophisticated understanding of the external world from its own experience through the adaptive, goal-oriented and incremental development of ever more complex skills and knowledge. 

Continual learning (CL) is essential in robotics where high dimensional data streams need to be constantly processed and where naïve continual learning strategies have been shown to suffer from _catastrophic forgetting_ also known as _catastrophic interference_. Take deep learning architectures for example: they excel at a number of classification and regression tasks by using a large dataset, however the dataset must be present during the training phase and the whole network must be _retrained_ every time there the underlying distribution $p_{data}$ changes. The existing approaches of _transfer learning_ that use weights of a different dataset to set the weights for the new dataset are inadequate, since they only help on reducing the training time for new dataset, rather than preserve the pre-existing ability to achieve good classification performance on the original dataset classes. Aside from supervised learning, CL accommodate a wider range of tasks such as unsupervised and reinforcement learning.  

Our brains exhibits _plasticity_ that allows us on one hand to integrate new knowledge and on the other hand to do so without overcompensating for new knowledge causing interference with the consolidated knowledge - this tradeoff is called _plasticity-stability dilemma_.  Hebbian plasticity, termed after Hebb's rule (1949), basically postulate the tendency of _strengthening the connection_ between pre-synaptic and post-synaptic neurons when the activations $x$ of the former affect the activations $y$ of the later. In its simplest form, Hebb's rule states that a synaptic strength $w$ changes as: $\Delta w = \eta × x \times y$ where $\eta$ is the learning rate. Hebb's rule can lead to instability and [homeostatic](https://en.wikipedia.org/wiki/Homeostasis) mechanisms are used, represented by a modulatory feedback control signal $m$ that regulates the unstable dynamics of Hebb's rule:

$$ \Delta w = m \times \eta × x \times y$$

We can draw the block diagram of such dynamical system model:

![Hebbian-Homeostatic-Plasticity](images/Hebbian-Homeostatic-Plasticity.png#center)
*Hebbian-Homeostatic Plasticity Model for synaptic updates.*

While the synaptic updates under the plasticity rule are essential to avoid catastrophic forgetting at the neural circuit level, we need a system level mechanism to carry out two complementary tasks, also known as Complementary Learning Systems (CLS) theory: _statistical learning_ with the ability to generalize across experiences and retain what it learned for the long term and _episodic learning_ that learns quickly novel information and retains episodic event memories (memorization). Statistical learning is implemented by the neocortical functional area of the grain and is a slower process as it extracts statistical structures of perceptive signals to be able to generalize to novel examples / situations. Episodic leanring, implemented in the hypocampus, is much faster as its goal is to learn specific events, retain them and play them back to the neocortex for information integration purposes. Both subsystems use the controlled Hebbian plasticity mechanism outlined earlier. 

![CLS](images/CLS.png#center)
*Hypocampus and Neocortex complementary learning system*

The aforementioned widely accepted as explanatory functions of the brain learning system, guided computational models and architectures of continuous learning in AI. For the deep learning architectures, three are the main threads: 

1. Imposing weight constraints (Regularization)
2. Allocating additional neural resources to capture the new knowledge (dynamic architectures)
3. CLS-based approaches. 

In this project, you are free to select a method from any of these three non-exclusive categories, [as described in detail](https://arxiv.org/abs/1802.07569). This is a [very active area of AI research](https://sites.google.com/view/clvision2020/challenge?authuser=0). 

## Datasets and Tasks

You are given two dataset options for this project as shown in the table below. The CORe50 option is more difficult than the MNIST option. Grading will happen relative to teams that selected the same option. The CORe50 option may require an AWS/Azure/Google cloud or NYU GPU compute  resource to run. MNIST should be able to run in Colab/Kaggle with standard free accounts. If you have access to compute, you will learn more from selecting the CORe50 option as this is closest to a real life dataset than the rotated MNIST which is not even testing for new classes - rather it tests CL on rotated version of classes it has seen before.

|CORe50 Option   | Rotated MNIST Option    |
| --- | --- |
|  You will use [this](https://vlomonaco.github.io/core50/index.html) dataset and evaluate your method for New Class (NC) scenario.   |  Use the dataset provided [here](https://github.com/facebookresearch/GradientEpisodicMemory)  based on [this paper](http://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf) |
|   Object recognition (classification).  | Object recognition (classification). |


