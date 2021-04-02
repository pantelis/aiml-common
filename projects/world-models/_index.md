---
title: World Models
---

# World Models - Can agents learn inside of their own dreams?

Humans build an internal mental model of the world and routinely use simulation to update it state. The work in [World Models](https://worldmodels.github.io/) uses RNNs to tackle Reinforcement Learning (RL) tasks and divide the agent into a substantial world model and a smaller controller model. 

![true-traj](images/true_traj.gif#center)

## Task 1 (20 points)

Read the paper and write your own 4-page report of the technique (a model-based RL), in a tutorial like fashion so computer scientists can still understand it.  

## Task 2 (40 points)

In this task you are asked to reproduce the results for the car racing environment. Consider adopting [this repo](https://github.com/zacwellmer/WorldModels) (30 points). 

Add to the report of a section that documents your own results and comment on the speed of learning a policy that drive the car around the track (10 points)
 
## Task 3 (40 points)

Familiarize yourself with Generative Adversarial Networks (GANs) and document a new approach that replaces the VAE with a GAN. (10 points) 
Repeat the experiment in the car racing environment. Add to the report what improvements (if any) you observed from the transition to GAN in terms of performance. (30 points)

### Notes: 

To help plan the tasks amongst the team members please use Trello (login with your nyu account). Submit your Trello boards with your project.

{{<hint warning "WARNING">}}

Note to all students using AWS resources provided for this project. You are **entirely** responsible to limit the use of p2.xlarge instances that consumes on-demand 1$/h and use CPU instances when possible. Invest in understanding docker containers so you do not suffer catastrophic failures when working in a cloud environment (the norm in practice).  You are responsible for not exceeding your budget.

{{</hint>}}