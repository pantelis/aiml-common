---
title: Lecture 9 - Reinforcement Learning
weight: 110
draft: false
---

# Reinforcement Learning

![unified-view-rl](images/unified-view-rl.png#center)
*Different Approaches to solve known and unknown MDPs*

We started looking at different agent behavior architectures starting from the [planning agents]({{<ref "../planning">}}) where the _model_ of the environment is known and with _no interaction_ with it the agent improves its policy, using this model as well as problem solving and logical reasoning skills. 

We then looked at agents that can plan by interacting with the environment still knowing the model - this was covered in the [MDP]({{<ref "../mdp">}}) chapter.  We have seen that DP uses _full width_ backups as every successor state and action is considered and evaluated using the known transition (environment dynamics) and reward functions. This can be dealt with for moderate size problems but even a single backup cant be feasible when we have very large state spaces. So we definitely need to develop approaches that allow agents to 

* optimally act in very large _known_ MDPs or 
* optimally act when we don't know the MDP functions. 

In this chapter we we outline _prediction_ and _control_ methods that are basic building blocks behind both problems. 

We develop agents that can act in an _initially unknown_ environment and learn via their _interactions_ with it, gradually improving their policy. In the reinforcement learning problem setting, agents _do not know_ essential elements of the MDP $\mathcal M = <\mathcal S, \mathcal P, \mathcal R, \mathcal A, \gamma>$ that were assumed as given in fully observed MDP. This includes the transition function, $P^a_{ss^\prime}$ and the reward function $\mathcal R_s^a$ that are essential as we have seen previously to estimate the value function and optimize the policy. 

The only way an agent can get information about these missing functions is through its experiences (states, actions, and rewards) in the environment—that is, the sequences of tuples ($S_t, A_t, R_{t+1}$).  

Provided that it can _learn_ such functions, RL can be posed as an MDP and many concepts we have already covered in the [MDP]({{<ref "../mdp">}}) chapter still apply. 
 
To scale to large problems however, we also need to develop approaches that can learn such functions efficiently both in terms of computation and space (memory). We will use DNNs to provide, in the form of approximations, the needed efficiency boost. 

![drl-concept](images/drl-concept.png#center)
*DRL principle - we will cover it in the SARSA section.*

Suffice to say that exploring DRL algorithms is a very long journey as shown below - we will cover only three key algorithms: REINFORCE, SARSA that can be used as design patterns for the others. These algorithms were not invented in vacuum though. The reader must appreciate that these algorithms are instantiations of the so called model-free prediction and model-free control approaches to solving either unknown MDP problems or known MDP problems that are too large to apply the methods outlined in the [MDP]({{<ref "../mdp">}}) chapter. 

![drl-algorithm-evolution](images/drl-algorithm-evolution.png#center)
*DRL algorithms - taxonomy and evolution*

{{<hint info >}}
> Apart from the notes here that are largely based on [David Silver's (Deep Mind) course material](https://www.davidsilver.uk/teaching/) and [video lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), the curious mind will be find additional resources: 
> * [in the Richard Sutton's book](http://incompleteideas.net/book/RLbook2020.pdf) - David Silver's slides and video lectures are based on this book. The code in Python of the book is [here](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
> * [in the suggested book](https://www.amazon.com/Deep-Reinforcement-Learning-Python-Hands-dp-0135172381/dp/0135172381/ref=mt_paperback?_encoding=UTF8&me=&qid=) written by Google researchers as well as on [OpenAI's website](https://openai.com/resources/). The chapters we covered is 1-4. 
> * You may also want to watch Andrew Ng's, [2018 version of his ML class](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) that includes MDP and RL lectures.
>  * The lecture  by [John Schulman of OpenAI](https://www.youtube.com/watch?v=PtAIh9KSnjo) is also very useful. 
{{</hint>}}