---
title: Dynamic Programming Algorithms - Policy Iteration
---

# Dynamic Programming Algorithms - Policy Iteration

In this section we develop dynamic programming algorithms for the so called _planning_ problem (which is RL without learning) where we are dealing with a _perfectly known MDP_. 

In the [Bellman expectation backup]({{<ref "../../bellman-expectation-backup">}}) section we have derived the equations which allowed us to efficiently compute the value function. 

We have seen also that the  [Bellman optimality backup]({{<ref "../../bellman-optimality-backup">}})  that are non linear and need to be solved using iterative approaches - their solution will result in the optimal value function $v_*$ and the optimal policy $\pi_*$. 

In policy iteration, given the policy $\pi$, we oscillate between two distinct steps as shown below:

![policy-iteration-steps](images/policy-iteration-summary.png#center)
*Policy iteration in solving the MDP - in each iteration we execute two steps, policy evaluation and policy improvement*

1. In the _policy evaluation_ (also called the _prediction_) step we estimate the state value function $v_\pi ~ \forall s \in \mathcal S$.

2. In the _policy improvement_ (also called the _control_) step we apply the greedy heuristic and elect a new policy based on the evaluation of the previous step. 

This is shown below,

![policy-iteration-convergence](images/policy-iteration-convergence.png#center)
*Policy and state value convergence to optimality in policy iteration. Up arrows are the evaluation steps while down arrows are the improvement steps.*

It can be shown that the policy iteration will converge to the optimal value function $v_*(s)$ and policy $\pi_*$. 

## A policy iteration example

The grid world example shown below is characterized by:

* Not discounted episodic MDP (γ = 1)
* Non terminal states 1, ..., 14
* One terminal state (shown twice as shaded squares)
* Actions leading out of the grid leave state unchanged
* Reward is −1 until the terminal state is reached
* Agent follows uniform random policy $\pi(north|.) = \pi(south|.) = \pi(east|.) = \pi(west | .) = 0.25$

The application of policy iteration to this problem results in:

![gridworld-policy-iterations](images/gridworld-policy-iterations.png#center)
*Convergence to optimal policy via separate prediction and policy improvement iterations*
