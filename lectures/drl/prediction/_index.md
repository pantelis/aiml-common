---
title: Model-free Prediction
---

# Model-free Prediction

In this chapter we find optimal policy solutions when the MDP is _unknown_ and we need to _learn_ its underlying value functions - also known as the  _model free_ prediction problem.  The main idea here is to learn value functions via sampling. These methods are in fact also applicable when the MDP is known but its models are simply too large to use the approaches outlined in the  [MDP chapter]({{<ref "../../mdp">}}). The two sampling approaches we will cover here are (incremental) Monte-Carlo (MC) and Temporal Difference (TD). 

**Notice that we started using capital letters for the _estimates_ of the value functions.**  

## Monte-Carlo (MC) Learning

The value functions $v_π$ and $q_π$ can be estimated from experience.  For example, if an agent follows policy $\pi$ and maintains an average, for each state encountered, of the actual returns that have followed that state, then the average will converge to the state’s value,$v_π(s)$, as the number of times that state is encountered approaches infinity.  If separate averages are kept for each action taken in each state, then these averages will similarly converge to the action values,$q_π(s,a)$.  We call estimation methods of this kind _Monte Carlo_ methods because they involve averaging over many random samples of returns.

In the MC learning approach more specifically, for every state at time $t$ we sample one complete trajectory (episode) as shown below.  

![mc-value-iteration-tree](images/mc-value-iteration-tree.png#center)
_Backup tree with value iteration based on the MC approach. MC samples a complete trajectory to the terminating node T shown with red._

There is some rationale of doing so, if we recall that the state-value function that was defined in the introductory [MDP section]({{<ref "../../mdp/mdp-intro">}}) i.e. the _expected_ return.

$$v_\pi(s) = \mathop{\mathbb{E}_\pi}(G_t | S_t=s)$$

can be approximated by using the _sample mean_ return over a _sample_ episode / trajectory:

$$G_t(\tau) = \sum_{k=0}^{T-1}\gamma^k R_{t+1+k}$$

The value function is therefore approximated in _Monte-Carlo_, by the sample mean of the returns over multiple episodes / trajectories. In other words, to update each element of the state value function 

1. For each time step $t$ that state $S_t$ is visited in an episode
   * Increment a counter $N(S_t)$ of visitations  
   * Calculate the total return $S(S_t) = S(S_t) + G_t$
2. At the end of multiple episodes, the value is estimated as $V(S_t) = S(S_t) / N(S_t)$

As $N(S_t) \rightarrow ∞$ the estimate will converge to $V(S_t) \rightarrow v_\pi(s)$. 

But we can also do the following trick, called _incremental mean approximation_: 

$$ \mu_k = \frac{1}{k} \sum_{j=1}^k x_j = \frac{1}{k} \left( x_k + \sum_{j=1}^{k-1} x_j \right)$$ 
$$ = \frac{1}{k} \left(x_k + (k-1) \mu_{k-1}) \right) =  \mu_{k-1} + \frac{1}{k} ( x_k - \mu_{k-1} )$$

Using the incremental sample mean, we can approximate the value function after each episode if for each state $S_t$ with return $G_t$,
{{<hint danger>}}

$$ N(S_t) = N(S_t) +1 $$
$$ V(S_t) = V(S_t) + \alpha \left( G_t - V(S_t) \right)$$

where $\alpha = \frac{1}{N(S_t)}$ can be interpreted as the forgetting factor. 
{{</hint>}}

$\alpha$ can also be any number $< 1$ to get into a more flexible sample mean - the _running mean_ that will increase the robustness of this approach in non-stationary environments.

## Temporal Difference (TD) Approximations

![td-value-iteration-tree](images/td-value-iteration-tree.png#center)
*Backup tree for value iteration with the TD approach. TD samples a single step ahead as shown with red.* 

Instead of getting an estimated value function at the end of multiple episodes, we can use the incremental mean approximation to update the value function after each step. 

Going back to the example of crossing the room optimally, we take one step towards the goal and the we  _bootstrap_ the value function of the state we were in from an estimated return for the remaining trajectory. We repeat this as we go along effectively adjusting the value estimate of the starting state from the true returns we have experienced up to now, gradually grounding the whole estimate as we approach the goal. 

![td-driving-to-work-example](images/td-driving-to-work-example.png#center)
*Two value approximation methods: MC (left), TD (right) as converging in their predictions of the value of each of the states in the x-axis. The example is from a hypothetical commute from office back home. In MC you have to wait until the episode ended (reach the goal) to update the value function at each state of the trajectory. In contrast, TD updates the value function at each state based on the estimates of the total travel time. The goal state is "arrive home", while the reward function is time.*

As you can notice in the figure above the solid arrows in the MC case, adjust the predicted value of each state to the _actual_ return while in the TD case the value prediction happens every step in the way. We call TD for this reason an _online_ learning scheme. Another characteristic of TD is that it does not depend on reaching the goal, it _continuously_ learns. MC does depend on the goal and therefore is _episodic_. This is important in many mission critical applications eg self-driving cars where you dont wait to "crash" to apply corrections to your state value based on what you experienced.

Mathematically, instead of using the _true_ return, $G_t$, something that it is possible in the MC as we are trully experiencing the world along a trajectory, TD uses a (biased) _estimated_ return called the _TD target_: $ R_{t+1} + \gamma V(S_{t+1})$ approximating the value function as:

{{<hint danger>}}

$$ V(S_t) = V(S_t) + \alpha \left( R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right)$$

{{</hint>}}

The difference below is called the _TD approximation error_,

$$\delta_t = R_{t+1} + \gamma (V(S_{t+1}) - V(S_t))$$

### The TD($\lambda$)

The TD approach of the previous section, can be extended to multiple steps. Instead of a single look ahead step we can take multiple successive look ahead steps (n), we will call this TD(n) for now, and at the end of the n-th step, we use the value function at that state to backup and get the value function at the state where we started. Effectively after n-steps our return will be:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{n-1}R_{t+n} + \gamma_n V(S_n)$$

and the TD(n) learning equation becomes

$$ V(S_t) = V(S_t) + \alpha \left( G^{(n)}_t - V(S_t) \right) $$

We now define the so called $\lambda$-return that combines all n-step return $G_t^{(n)}$ via the weighting function shown below as,

$$G_t^{(n)} = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

![lambda-weighting-function](images/lambda-weighting-function.png#center)
*$\lambda$ weighting function for TD($\lambda$)*

the TD(n) learning equation becomes

{{<hint danger>}}

$$ V(S_t) = V(S_t) + \alpha \left( G^\lambda_t - V(S_t) \right) $$

{{</hint>}}

When $\lambda=0$ we get TD(0) learning, while when $\lambda=1$ we get learning that is roughly equivalent to MC. It is instructive to see the difference between MC and TD approaches. 

![td-vs-mc](images/td-vs-mc.png#center)
*TD vs MC approaches to $V(s)$ function estimation. TD is converging much faster but it has larger bias compared to MC.* 