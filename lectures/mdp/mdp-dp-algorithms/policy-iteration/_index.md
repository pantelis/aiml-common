---
title: Dynamic Programming Algorithms - Policy Iteration
---

# Dynamic Programming Algorithms - Policy Iteration

In this section we develop dynamic programming algorithms for the so called _planning_ problem (which is RL without learning) where we are dealing with a _perfectly known MDP_. 

In the [Bellman expectation backup]({{<ref "../../bellman-expectation-backup">}}) section we have derived the equations which allowed us to efficiently compute the value function. 

We have seen also that the  [Bellman optimality backup]({{<ref "../../bellman-optimality-backup">}})  that are non linear and need to be solved using iterative approaches - their solution will result in the optimal value function $v_*$ and the optimal policy $\pi_*$. Since the Bellman equations allow us to decompose recursively the problem into sub-problems, they in fact implement a general and exact approach called _dynamic programming_ which assumes full knowledge of the MDP.

In policy iteration, given the policy $\pi$, we iterate two distinct steps as shown below:

![policy-iteration-steps](images/policy-iteration-summary.png#center)
*Policy iteration in solving the MDP - in each iteration we execute two steps, policy evaluation and policy improvement*

1. In the _evaluation_ (also called the _prediction_) step we estimate the state value function $v_\pi ~ \forall s \in \mathcal S$.

2. In the _improvement_ (also called the _control_) step we apply the greedy heuristic and elect a new policy based on the evaluation of the previous step. 

This is shown next,

![policy-iteration-convergence](images/policy-iteration-convergence.png#center)
*Policy and state value convergence to optimality in policy iteration. Up arrows are the evaluation steps while down arrows are the improvement steps.*

It can be shown that the policy iteration will converge to the optimal value function $v_*(s)$ and policy $\pi_*$. 

#### Policy Evaluation Step

The policy $\pi$ is evaluated when we have produced the state-value function $v_\pi(s)$ for all states. In other words when we know the expected discounted returns that each state can offer us. To do so we apply the Bellman expectation backup equations repeatedly in an iterative fashion. 

We start at $k=0$ by initializing all state-value function (a vactor) to $v_0(s)=0$. In each iteration $k+1$ we start with the state value function of the previous iteration $v_k(s)$ and apply the Bellman expectation backup as prescribed by the one step lookahead tree below that is decorated relative to what [we have seen]({{<ref "../../bellman-expectation-backup">}}) with the iteration information. This is called the synchronous backup formulation as we are updating all the elements of the value function vector at the same time. 

![policy-evaluation-tree](images/policy-evaluation-tree.png#center)
*Tree representation of the state-value function with one step look ahead across iterations.*

The Bellman expectation backup is given by,

$$v_{k+1}(s) = \sum_{a \in \mathcal A} \pi(a|s) \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_k(s^\prime) \right)$$

and in vector form,

$$\mathbf{v}^{k+1} = \mathbf{\mathcal R}^\pi + \gamma \mathbf{\mathcal P}^\pi \mathbf{v}^k$$

The [following source code](https://becomesentient.com/mdp-dynamic-programming/) is instructive and standalone. It executes the policy evaluation for the Gridworld environment from the many that are part of the Gym RL python library. 
 
{{< details "Policy Evaluation Python Code" "..." >}}

```python
# this code is from https://becomesentient.com/mdp-dynamic-programming/

import numpy as np 
import gym.spaces
from gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, epsilon=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V_old = np.zeros(env.nS)

    while True:
        
        #new value function
        V_new = np.zeros(env.nS)
        #stopping condition
        delta = 0

        #loop over state space
        for s in range(env.nS):

            #To accumelate bellman expectation eqn
            v_fn = 0
            #get probability distribution over actions
            action_probs = policy[s]

            #loop over possible actions
            for a in range(env.nA):

                #get transitions
                [(prob, next_state, reward, done)] = env.P[s][a]
                #apply bellman expectatoin eqn
                v_fn += action_probs[a] * (reward + discount_factor * V_old[next_state])

            #get the biggest difference over state space
            delta = max(delta, abs(v_fn - V_old[s]))

            #update state-value
            V_new[s] = v_fn

        #the new value function
        V_old = V_new

        #if true value function
        if(delta < epsilon):
            break

    return np.array(V_old)


random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

print(v)
print(expected_v)
```
{{</details>}}

#### Policy Improvement Step

In the policy iteration step we are given the value function and simply apply the greedy heuristic to it.

$$\pi^\prime = \mathtt{greedy}(v_\pi)$$

It can be shown that this heuristic results into a policy that is better than the one the prediction step started ($\pi^\prime \ge \pi$) and this extends into multiple iterations. We can therefore converge into an optimal policy - the interested reader can follow [this](https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=3) lecture for a justification. 

{{< details "Policy Iteration Python Code" "..." >}}

```python
import numpy as np
import gym.spaces
from gridworld import GridworldEnv

env = GridworldEnv()

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
       
    """
    def one_step_lookahead(s, value_fn):

        actions = np.zeros(env.nA)

        for a in range(env.nA):

            [(prob, next_state, reward, done)] = env.P[s][a]
            actions[a] = prob * (reward + discount_factor * value_fn[next_state])
            
        return actions

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    actions_values = np.zeros(env.nA)

    while True:

        #evaluate the current policy
        value_fn = policy_eval_fn(policy, env)
       
        policy_stable = True

        #loop over state space
        for s in range(env.nS):

            #perform one step lookahead
            actions_values = one_step_lookahead(s, value_fn)
            
        	#maximize over possible actions 
            best_action = np.argmax(actions_values)

            #best action on current policy
            chosen_action = np.argmax(policy[s])

    		#if Bellman optimality equation not satisifed
            if(best_action != chosen_action):
                policy_stable = False

            #the new policy after acting greedily w.r.t value function
            policy[s] = np.eye(env.nA)[best_action]

        #if Bellman optimality eqn is satisfied
        if(policy_stable):
            return policy, value_fn

    
    

policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
```
{{</details>}}

We already have seen that in the Gridworld environment, we may not _need_ to reach the optimal state value function $v_*(s)$ for an optimal policy to result, as shown in the next figure where the value function for the $k=3$ iteration results the same policy as the policy from a far more accurate value function (large k). The gridworld below is characterized by:

* Not discounted episodic MDP (γ = 1)
* Non terminal states 1, ..., 14
* One terminal state (shown twice as shaded squares)
* Actions leading out of the grid leave state unchanged
* Reward is −1 until the terminal state is reached
* Agent follows uniform random policy $\pi(north|.) = \pi(south|.) = \pi(east|.) = \pi(west | .) = 0.25$

![gridworld-policy-iterations](images/gridworld-policy-iterations.png#center)
*Convergence to optimal policy via separate prediction and policy improvement iterations*

We can therefore stop early and taking the argument to the limit, do the policy improvement step in _each_ iteration. 

In summary, we have seen that policy iteration solves the known MDPs. In the next section we remove the known MDP assumption and deal with the first Reinforcement Learning (RL) algorithm. 

