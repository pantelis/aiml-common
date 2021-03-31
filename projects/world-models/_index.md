---
title: World Models
---

# World Models - Can agents learn inside of their own dreams?

Humans build an internal mental model of the world and routinely use simulation to update it state. The work in [World Models](https://worldmodels.github.io/) uses RNNs to tackle Reinforcement Learning (RL) tasks and divide the agent into a substantial world model and a smaller controller model. 

![true-traj](images/true_traj.gif#center)

## Task 1 (20 points)

Read the paper and write your own 4-page summary of the technique (a model-based RL), in a tutorial like fashion so computer scientists can still understand it.  

## Task 2 (30 points)

In this task you are asked to reproduce the results for the car racing environment. Consider adopting [this repo](https://github.com/zacwellmer/WorldModels). 
Document your own results and comment on the speed of learning a policy that drive the car around the track.
 
## Task 3 (30 points)

Repeat Task 2 but now replace RNN with a CNN of your choice (eg ResNet). 