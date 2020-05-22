---
title: Model-free Control
weight: 106
draft: false
---

# Model-free Control

In this section we outline methods that can result in optimal policies when the MDP is _unknown_ and we need to _learn_ its underlying functions / models - also known as the  _model free_ control problem. Learning in this chapter, follows the _on-policy_ approach where the agent learns these models "on the job", so to speak. 

In the [model-free prediction]({{<ref "../prediction">}}) section we have seen that it is in fact possible to get to estimate the state value function without any MDP model dependencies. However, when we try to do do greedy policy improvement 

$$\pi^\prime(s) = \argmax_{a \in \mathcal A} (\mathcal R_s^a + \mathcal P_{ss^\prime}^a V(s^\prime))$$

we do have dependencies on knowing the dynamics of the MDP. So the obvious question is - have we done [all this discussion]({{<ref "../prediction">}}) in vain? It turns out that we did not. All it takes is to **apply prediction to the state-action value function $Q(s,a)$** and then apply the greedy policy improvement step

$$\pi^\prime = \argmax_{a \in \mathcal A} Q(s,a)$$

This is shown next: 

![generalized-policy-iteration](images/generalized-policy-iteration.png#center)
*Generalized policy iteration using action-value function*

The following tables summarize the relationship between TD backup and DP (full backup). The entry for [SARSA]({{<ref "../sarsa">}}) will be evident after you go through it. 

![dp-td-tree-comparison](images/dp-td-tree-comparison.png#center)
*Backup Trees for DP vs TD*

![dp-td-comparison-equations](images/dp-td-comparison-equations.png#center)
*Update equation comparison between DP vs TD*
