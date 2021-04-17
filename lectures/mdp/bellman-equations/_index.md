---
title: Bellman equations 
---

# Bellman equations


## Computing the value functions given a policy

In this section we describe how to calculate the value functions by establishing a recursive relationship similar to the one we did for the return. We replace the expectations with summations over quantities such as states and actions. 

Lets start with the state-value function that can be written as, 

$$v(s) = \mathop{\mathbb{E}} \left[G_t | S_t=s\right] = \mathop{\mathbb{E}} \left[ \sum_{k=0}^∞\gamma^k R_{t+1+k} | S_t=s\right]$$

$$ = \mathop{\mathbb{E}} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+ ... | S_t=s \right]$$

$$ = \mathop{\mathbb{E}} \left[ R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3}+ ...) | S_t=s \right]$$

$$ = \mathop{\mathbb{E}} \left[ R_{t+1} + \gamma v(S_{t+1}=s^\prime) | S_t=s \right]$$

NOTE: All above expectations are with respect to policy $\pi$.

This is perhaps one of the _most important_ recursions in control theory - it is known as the **Bellman expectation equation** repeated below:

$$v_\pi(s) = \mathop{\mathbb{E}_\pi} \left[ R_{t+1} + \gamma ~ v_\pi(S_{t+1}=s^\prime) | S_t=s \right]$$

The parts of the value function above are: 

1. The immediate reward, 
2. The discounted value of the successor state $\gamma v(S_{t+1}=s^\prime)$.

Similarly to the state-value function we can decompose the action-value function as,

$$q_\pi(s,a) = \mathop{\mathbb{E}_\pi} \left[ R_{t+1} + \gamma ~ q_\pi(S_{t+1}=s^\prime, A_{t+1}) | S_t=s, A_t=a \right]$$

We now face the problem that we need to compute these two value functions and we start by considering what is happening at each time step. At each time step while in state $S_t=s$ we have a number of actions we can choose, the probabilities of which depend on the policy $\pi(a|s)$. What value we can reap from each action is given to us by $q_\pi(s,a)$.  This is depicted below. 

![state-value-tree](images/state-value-tree.png#center)
*Actions can be taken from that state $s$ according to the policy $\pi$. Actions are represented in this simple tree with action nodes (solid circles) while state nodes are represented by empty circles.*

Translating what we have just described in equation form, allows us to write the state-value equation as,

$$v(s) = \sum_{a \in \mathcal A} \pi(a|s) q_\pi(s,a)$$

This sum is easily understood if you move backwards from the action nodes of the tree to the state node. Each edge weighs with $\pi(a|s)$ the corresponding action-value. This backwards calculation is referred to as a _backup_. 

We can now reason fairly similarly about the action-value function that can be written by taking the expectation,

$$q_\pi(s,a)  = \mathop{\mathbb{E}_\pi} \left[ R_{t+1} |  S_t=s, A_t= a \right] + \gamma ~ \mathop{\mathbb{E}_\pi} \left[ v_\pi(S_{t+1}=s^\prime) | S_t=s, A_t= a \right]$$

The first expectation is the reward function $\mathcal{R}^a_s$ by definition. The second expectation can be written in matrix form by considering that at each time step if we are to take an action $A_t=a$, the environment can transition to a number of successor states $S_{t+1}=s'$ and signal a reward $R_{t+1}$ as shown in the next figure. 

![action-value-tree](images/action-value-tree.png#center)
_Successor states that can be reached from state $s$ if the agent selects action $a$. $R_{t+1}  = r$ is the instantaneous reward for each of the possibilities._

If you recall the agent in the Gridworld, has 80% probability to achieve its intention and make the environment to change the desired state and 20% to make the environment change to not desired states justifying the multiplicity of states given an action in the figure above. 

What successor states we will transition to depends on the _transition model_ $P^a_{ss^\prime} = p[S_{t+1}=s^\prime | S_t=s, A_t=a ]$ . What value we can reap from each successor state is given by $v_\pi(s^\prime)$. The expectation can then be evaluated as a summation over all possible states $\sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v(s^\prime)$. In conclusion, the action-value function can be written as

$$q_\pi(s,a) = \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_\pi(s^\prime)$$

Substituting the  $v_\pi(s^\prime)$ is represented by the following tree that considers the action-value function over a look ahead step. 

![action-state-action-value-tree](images/action-state-action-value-tree.png#center)
*Tree that represents the action-value function after a one-step look ahead.*

{{<hint danger>}}
$$q_\pi(s,a) = \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} \sum_{a^\prime \in \mathcal A} \pi(a^\prime|s^\prime) q_\pi(s^\prime,a^\prime)$$
{{</hint>}}

Now that we have a computable $q_\pi(s,a)$ value function we can go back and substitute it into the equation of the state-value function. Again we can representing this substitution by the tree structure below.

![state-action-state-value-tree](images/state-action-state-value-tree.png#center)
*Tree that represents the state-value function after a one-step look ahead.*

With the substitution we can write the state-value function as,

{{<hint danger>}}
$$v_\pi(s) = \sum_{a \in \mathcal A} \pi(a|s) \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_\pi(s^\prime) \right)$$
{{</hint>}}

As we will see in a separate chapter, this equation is going to be used to iteratively calculate the converged value function of each state given an MDP and a policy.  The equation is referred to as the _Bellman expectation backup_ - it took its name from the previously shown tree like structure where we use state value functions from the leaf modes $s^\prime$ to the root node.

### Solving the MDP

Now that we can calculate the value functions efficiently via the Bellman expectation recursions, we can now solve the MDP which requires maximize either of the two functions over all possible policies.  The _optimal_ state-value function and _optimal_ action-value function are given by definition,

$$v_*(s) = \max_\pi v_\pi(s)$$
$$q_*(s,a) = \max_\pi q_\pi(s,a)$$

If we can calculate $q_*(s,a)$ we have found the best possible action in each state of the environment. In other words we can now obtain the _optimal policy_ by maximizing over $q_*(s,a)$ - mathematically this can be expressed as,

$$\pi_*(a|s) = \begin{cases}1 & \text{if }\ a = \argmax_{a \in \mathcal A} q_*(s,a), \\\\ 
0 & \text{otherwise}\end{cases}$$

So the problem now becomes how to calculate the optimal value functions. We return to the tree structures that helped us understand the interdependencies between the two and this time we look at the optimal equivalents. 

![optimal-state-value-tree](images/optimal-state-value-tree.png#center)
*Actions can be taken from that state $s$ according to the policy $\pi_*$*

Following similar reasoning as in the Bellman expectation equation where we calculated the value of state $s$ as an average of the values that can be claimed by taking all possible actions, now we simply replace the average with the max. 

$$v_*(s) = \max_a q_*(s,a)$$

![optimal-action-value-tree](images/optimal-action-value-tree.png#center)
*Successor states that can be reached from state $s$ if the agent selects action $a$. $R_{t+1}=r$ we denote the instantaneous reward for each of the possibilities.*

Similarly, we can express  $q_*(s,a)$ as a function of $v_*(s)$ by looking at the corresponding tree above. 

$$q_*(s,a) = \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_*(s^\prime)$$

Notice that there is no $\max$ is this expression as we have no control on the successor state - that is something the environment controls. So all we can do is average. 

Now we can similarly attempt to create a _recursion_ that will lead to the **Bellman optimality equations** that effectively solve the MDP, by expanding the trees above.

![optimal-state-action-state-value-tree](images/optimal-state-action-state-value-tree.png#center)
*Tree that represents the optimal state-value function after a two-step look ahead.*

![optimal-action-state-action-value-tree](images/optimal-action-state-action-value-tree.png#center)
*Tree that represents the optimal action-value function after a two-step look ahead.*

{{<hint danger>}}
$$v_*(s) = \max_a \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_*(s^\prime) \right)$$

$$q_*(s,a) = \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} \max_{a^\prime} q_*(s^\prime,a^\prime)$$
{{</hint>}}

These equations due to the $\max$ operator are non-linear and can be solved to obtain the MDP solution aka $q_*(s,a)$ iteratively via a number of methods: policy iteration, value iteration, Q-learning, SARSA. We will see some of these methods in detail in later chapters. The key advantage in the Bellman optimality equations is efficiency: 

1. They _recursively decompose_ the problem into two sub-problems: the subproblem of the next step and the optimal value function in all subsequent steps of the trajectory.
2. They cache the optimal value functions to the sub-problems and by doing so we can reuse them as needed.


