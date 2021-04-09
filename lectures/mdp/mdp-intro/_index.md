---
title: Introduction to MDP
weight: 102
draft: false
---

# Introduction to MDP

## The MDP Agent - Environment Interface 

We start by reviewing the agent-environment interface with this evolved notation and provide additional definitions that will help in grasping the concepts behind DRL. We treat MDP analytically effectively deriving the four Bellman equations 

![agent-env-interface](images/agent-env-interface.png#center)
*Agent-Environment Interface*

The following table summarizes the notation and contains useful definitions that we will use to describe required concepts later.  

| Symbol  | Description  |
|:-:|---|
| $A_t$ | agent action at time step $t$, $a \in \mathcal{A}$ the finite set of actions |
| $S_t$ | environment state at time step $t$, $s \in \mathcal{S}$ the finite set of states|
| $R_{t+1}$ | reward sent by the environment after taking action $A_t$ |
| $t$ | time step index associated with each tuple ($S_t, A_t, R_{t+1}$) called the *experience*. | 
| $T$ | maximum time step beyond which the interaction terminates |
| _episode_ | the time horizon from $t=0$ to $T-1$ |
| $\tau$ | _trajectory_ - the sequence of experiences over an episode |
| $G_t$ | _return_ - the total discounted rewards from time step $t$ - it will be qualified shortly. |
| $\gamma$ | the discount factor $\gamma \in [0,1]$ |

As the figure above indicates, the agent *perceives fully* the environment state $S_t$  (**fully observed**) via a bank of sensors. In other words the agent knows which state the environment is, perfectly. 

The agent function, called **policy** $\pi$, produces an action either deterministically $a=\pi(s)$ or stochastically where the function produces a probability of actions conditioned in the current state:

$$\pi(a|s)=p(A_t=a|S_t=s)$$

The policy is assumed to be stationary i.e. not change with time step $t$ and it will depend only on the state $S_t$ i.e. $A_t \sim \pi(.|S_t), \forall t > 0$

This will have two effects:

### State transition

The first is that the action itself will change the environment state to some other state. This can be represented via the environment _state transition_ probabilistic model that generically can be written as:

$$s_{t+1} = p( s_{t+1} | (s_0, a_0), ..., (s_t, a_t) )$$ 

Under the assumption that the next state only depends on the current state and action 

$$s_{t+1} = p( s_{t+1} | (s_0, a_0), ..., (s_t, a_t) ) =  p( s_{t+1} | s_t, a_t)$$ 

we define a Markov Decision Process as the 5-tuple $\mathcal M = <\mathcal S, \mathcal P, \mathcal R, \mathcal A, \gamma>$ that produces a sequence of experiences $(S_1, A_1, R_2), (S_2, A_2, R_3), ...$. Together with the policy $\pi$, the state transition probability matrix $\mathcal P$ is defined as

$$\mathcal P^a_{ss^\prime} = p[S_{t+1}=s^\prime | S_t=s, A_t=a ]$$

where $s^\prime$ simply translates in English to the successor state whatever the new state is.

{{<hint info>}}
Can you determine the state transition matrix for the 4x3 Gridworld in [MDP slides]({{<ref "../mdp-slides">}})?  What each row of this matrix represents?
{{</hint>}}

Note that Markov processes are sometimes erroneously called _memoryless_ but in any MDP above we can incorporate memory aka dependence in more than one state over time by cleverly defining the state $S_t$ as a container of a number of states. For example, $S_t = \left[ S_t=s, S_{t-1} = s^\prime \right]$ can still define an Markov transition using $S$ states. The transition model $p(S_t | S_{t-1}) = p(s_t, s_{t-1} | s_{t-1}, s_{t-2}) = p(s_t|s_{t-1}, s_{t-2})$ is called the 2nd order Markov chain. 

### Reward function

The second effect from the action, is that it will cause the environment to send the agent a signal called _instantaneous reward_ $R_{t+1}$. Please note that in the literature the reward is also denoted as $R_{t}$ - this is a convention issue rather than something fundamental. The justification of the index $t+1$ is that the environment will take one step to respond to what it receives from the agent. 

The _reward function_ tells us if we are in state $s$, what reward  $R_{t+1}$ in expectation we get when taking an action $a$. It is given by,

$$\mathcal{R}^a_s = \mathop{\mathbb{E}}[ R_{t+1} | S_t=s, A_t=a]$$ 

and it will be used later on during the development of value iteration. 

Each reward and state transition will trigger a new iteration and the interaction will terminate at some point either because the environment terminated after reaching a maximum time step ($T$) or reaching the goal.

## Value Functions

To define the value function formally, consider first the _return_ defined as the total discounted reward at time step $t$. 

$$G_t = R_{t+1} + \gamma R_{t+2} + ... = \sum_{k=0}^∞\gamma^k R_{t+1+k}$$

Notice the two indices needed for its definition - one is the time step $t$ that manifests where we are in the trajectory and the second index $k$ is used to index future rewards up to infinity - this is the case of infinite horizon problems. If the discount factor $\gamma < 1$ and there the rewards are bounded to $|R| < R_{max}$ then the above sum is _finite_. 

$$ \sum_{k=0}^∞\gamma^k R_{t+1+k} <  \sum_{k=0}^∞\gamma^k R_{max} = \frac{R_{max}}{1-\gamma}$$

The return is itself a random variable - for each trajectory defined by sampling the policy (strategy) of the agent we get a different return. For the Gridworld of the [MDP slides]({{<ref "../mdp-slides">}}):

$$\tau_1: S_0=s_{11}, S_1 = s_{12},  ... S_T=s_{43} \rightarrow G^{\tau_1}_0 = 5.6$$
$$\tau_2: S_0=s_{11}, S_1=s_{21}, ... , S_T=s_{43} \rightarrow G^{\tau_2}_0 = 6.9$$
$$ … $$

Please note that the actual values are different - these are sample numbers to make the point that the return depends on the specific trajectory.

The _state-value function_ $v_\pi(s)$ provides a notion of the long-term value of state $s$. It is equivalent to the _utility_ we have seen in the [MDP slides]({{<ref "../mdp-slides">}}). It is defined as the _expected_ return starting at state $s$ and following policy $\pi(a|s)$, 

$$v_\pi(s) = \mathop{\mathbb{E}_\pi}(G_t | S_t=s)$$

The expectation is obviously due to the fact that $G_t$ are random variables since the sequence of states of each trajectory is dictated by the stochastic policy. As an example, assuming that there are just two trajectories whose returns were calculated above, the value function of state $s_{11}$ will be

$$v_\pi(s_{11}) = \frac{1}{2}(G^{\tau_1}_0 + G^{\tau_2}_0)$$

One corner case is interesting - if we make $\gamma=0$ then $v_\pi(s)$  becomes the average of instantaneous rewards we can get from that state.

We also define the _action-value function_ $q_\pi(s,a)$ as the expected return starting from the state $s$, taking action $a$ and following policy $\pi(a|s)$.

$$q_\pi(s,a) = \mathop{\mathbb{E}_\pi} (G_t | S_t=s, A_t=a)$$

This is an important quantity as it helps us decide the action we need to take while in state $s$. 

### Computing the value functions given a policy

In this section we describe how to calculate the value functions. As you can imagine this means replacing the expectations with summations over quantities such as states and actions while, at the same time, making the required computations as efficient as possible. 

Lets start with the state-value function that can be written as, 

$$v(s) = \mathop{\mathbb{E}} \left[G_t | S_t=s\right] = \mathop{\mathbb{E}} \left[ \sum_{k=0}^∞\gamma^k R_{t+1+k} | S_t=s\right]$$
$$ = \mathop{\mathbb{E}} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+ ... | S_t=s \right]$$
$$ = \mathop{\mathbb{E}} \left[ R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3}+ ...) | S_t=s \right]$$
$$ = \mathop{\mathbb{E}} \left[ R_{t+1} + \gamma v(S_{t+1}=s^\prime) | S_t=s \right]$$

NOTE: All above expectations are with respect to policy $\pi$.

This is perhaps one of the _most important_ recursions in control theory - it is known as the **Bellman expectation equation** repeated below:

$$v_\pi(s) = \mathop{\mathbb{E}_\pi} \left[ R_{t+1} + \gamma ~ v_\pi(S_{t+1}=s^\prime) | S_t=s \right]$$

The parts of the value function above are (1) the immediate reward, (2) the discounted value of the successor state $\gamma v(S_{t+1}=s^\prime)$. Similarly to the state-value function we can decompose the action-value function as,

$$q_\pi(s,a) = \mathop{\mathbb{E}_\pi} \left[ R_{t+1} + \gamma ~ q_\pi(S_{t+1}=s^\prime, A_{t+1}) | S_t=s, A_t=a \right]$$

We now face the problem that we need to compute these two value functions and we start by considering what is happening at each time step. At each time step while in state $S_t=s$ we have a number of actions we can choose, the probabilities of which depend on the policy $\pi(a|s)$. What value we can reap from each action is given to us by $q_\pi(s,a)$.  This is depicted below. 

![state-value-tree](images/state-value-tree.png#center)
*Actions can be taken from that state $s$ according to the policy $\pi$. Actions are represented in this simple tree with action nodes (solid circles) while state nodes are represented by empty circles.*

Translating what we have just described in equation form, allows us to write the state-value equation as,

$$v(s) = \sum_{a \in \mathcal A} \pi(a|s) q_\pi(s,a)$$

This sum is easily understood if you move backwards from the action nodes of the tree to the state node. Each edge weighs with $\pi(a|s)$ the corresponding action-value. This backwards calculation is referred to a a _backup_. We can now reason fairly similarly about the action-value function that can be written by taking the expectation,

$$q_\pi(s,a)  = \mathop{\mathbb{E}_\pi} \left[ R_{t+1} |  S_t=s, A_t= a \right] + \gamma ~ \mathop{\mathbb{E}_\pi} \left[ v_\pi(S_{t+1}=s^\prime) | S_t=s, A_t= a \right]$$

The first expectation is the reward function $\mathcal{R}^a_s$ by definition. The second expectation can be written in matrix form by considering that at each time step if we are to take an action $A_t=a$, the environment can transition to a number of successor states $S_{t+1}=s'$ and signal a reward $R_{t+1}$ as shown in the next figure. 

![action-value-tree](images/action-value-tree.png#center)
*Successor states that can be reached from state $s$ if the agent selects action $a$. $R_{t+1}=r$ we denote the instantaneous reward for each of the possibilities.*

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


