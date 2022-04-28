
# PART 4/4: Reinforcement Learning (RL)

NOTE: Reinforcement Learning and Deep RL (DRL) are covered in Lecture 9. 



### Introduction

* Reinforcement learning is concerned with solving sequential decision problems.
* Many real world problems fall into this category e.g. playing video games, driving, robotic control etc.
* **_Reinforcement learning might be considered to encompass all of AI: an agent is placed in an environment and must learn to behave therein_**
* The goal of reinforcement learning is to use observed rewards to learn the optimal (or close to optimal) policy for the environment.
* So far we have been looking at solving sequential decision making problems but we have assumed a complete model of the environment  and the reward function
* Can we **learn** directly from experiences in the world?
	* Must receive feedback for good/bad experiences
	* Called rewards or **reinforcement** 
	* One assumption that we make is that the reward input is known i.e. we know that a particular sensory input corresponds to reward





### Learning

* An agent is learning if it improves its performance on future tasks after making observations about the world.
* What is learned:
	 * mapping from state to action
	 * utility information indicating desirability of states
	 * action-value information about the desirability of actions
	 * goals that describe states whose achievement maximize agent's utility
* Feeback types used that determine three types of learning 
	*  observes *patterns* in input without explicit feedback - unsupervised learning
	* *reinforcements* i.e. rewards or punishments - reinforcement learning
	* observes example input-output pairs and learns the mapping functions - supervised learning



### Utility, Value, Action-Value, Action-Utility, Q functions

* **Utility function**: $U^{\pi}(s)$ provide expected utility or return or reward of a state by executing a given policy $\pi$
* **Value function**: $V^{\pi}=U^{\pi}(s)$  
* **Action-utility function**: $U^{\pi}(s,a)$ gives the expected utility by taking an action $a$ while in state $s$
* **Action-value function**: $V^{\pi}(s,a)=U^{\pi}(s,a)$
* **Q-function**: $Q^{\pi}(s,a)=V^{\pi}(s,a)=U^{\pi}(s,a)$
* You get the value function by taking the expection of the action-value function over the set of actions i.e. $V^{\pi}(s) = \underset{a}{E}[Q(s,a)]$




### Reinforcement Learning 
* Reinforcement learning is learning what to do i.e. how to map situations to actions so as to maximize a numerical reward signal. The learner must discover which actions yield the most reward by trying them. 
* These two characteristics: **trial\-and\-error** search and **delayed reward** are the two most important distinguishing features of reinforcement learning.
* There is a feedback control loop where agent and environment exchange signals while the agent tries to maximize the rewards or objective.
* Signal exchanged at any time $t$ is $(s_t,a_t,r_t)$  which correspond to the state, action and reward at time $t$. This tuple is called an _experience_.

![](./images/agent-env-interface.png)<!-- .element width="450px" -->




### Reinforcement Learning as MDP

* Reinforcement learning can be formulated as an MDP with the following parameters:
	* transition function ( $ P(s_{t+1} | s_t,a_t) $ ) captures how the enviroment transitions from one state to the next and is formulated as MDP
	* reward function  ( $ R(s_t,a_t,a_{t+1}) $ )
	* set of actions $ A $
	* set of states $ S $ 
* One important assumption in the above formulation is that the agent does not have access to the transition or reward function 
* Functions to be learning in RL
	* policy $\pi$ 
	* value function $V^{\pi}$  or action value function $Q^{\pi}(s,a)$
	* environment model $P(s^{'} | s,a)$



### Deep RL
* use neural networks as function approximators to learn the functions
* **Policy-based methods**
	* learn policy $\pi$ to maximize objective
	* PROS
		* general class of optimization methods
		* any type of actions: discrete, continous or a mix
		* guaranteeed to converge(locally) for e.g. via Poilcy Gradient Algorithm
	* CONS
		* high variance
		* sample inefficient



### Deep RL - **Value Based Methods** 

* agent learns either $V^{\pi}(s)$ or $Q^{\pi}$
* uses the learnt function(s) to generate the policy for e.g. greedy policy
* generally $Q^{\pi}(s,a)$ is preferred as agent can select the action when in a given state to maximize the objective
* PROS:
	* more sample efficient than policy based algorithms
* CONS
	* no guarantee of convergence to optimal policy
	* most methods are for discrete action spaces though recently QT-OPT has been proposed which can handle continuous spaces



### Deep RL - **Model Based Methods** 

* agent learns a model of the environment dynamics
* using this model, agent can imagine or predict what will happen if a set of actions are taken for few time steps without actually changing the environment
* Based on these predictions, agent can figure out the best actions
* PROS:
	* gives agent foresight
	* tend to require fewer samples
* CONS:
	* learning the model can be difficult as typically real world environments can have a large state and action space
	* predictions depend on the accuracy of the model 



### Taxonomy of RL algorithms


![Overview of RL algorithms](./images/rl-taxonomy.png)<!-- .element width="800px" -->




### Off policy versus on policy

* On policy
	* e.g. SARSA
	* agent learns on the policy i.e. training data generated from the current poicy is used
	* in other words agent is learning the value of the policy that is being followed 
	* after the agent is trained, data is discarded and the iterated policy is used.
	* sample inefficent due to the discarding of data but memory efficient
* Off policy
	* e.g. Q-learning
	* any data collected can be used for training 
	* the value of a policy that is different from the one being followed is being learnt
	* more memory may be required to store data




### Deep Learning 

* Neural networks learn functions $f_{\theta}(x)$ which map input $x$ to output $y$. The weights of the neural network are represented by $\theta$.
	* if $y=f(x)$ then learnt $f_{\theta}(x)$ is the estimate for $f(x)$
	* Loss function $L(f_{\theta}(x),y)$ captures the difference between the target $y$ and the predicted network output $f_{\theta}(x)$. This loss function needs to be minimized.
	* generally we have training data samples which are independent and identically distributed (iid)
* Changing the weights will corresponding to different mapping functions.
* Increasing number of nodes in a layer or layers allows learning of more complex functions
* In reinforcement learning: 
	* neither $x$ or $y$ are known in advance
	* instead these values are obtained through agent interactions with environment - where it observes states and rewards
	* reward functions are the main source of feedback and the rewards are quite sparse
	* since current state and actions that an agent takes affect the future states, the iid assumption between samples for neural network training no longer holds and this affects the rate of convergence.


