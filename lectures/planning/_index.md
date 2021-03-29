---
title: Planning
---

# Planning

In [recursive state estimation]({{<ref "../pgm/recursive-state-estimation">}}) chapter we made two advances in our modeling tool set:

1. We introduced sequenced over time events and the concept of a varying _state_ over such sequences.  
2. We saw how the agent state as dictated by an underlying dynamical model and and how to estimate it recursively using a graphical model that introduced the Bayesian filter. We saw that many well known estimation algorithms such as the Kalman filter are specific cases of this framework. 

With this probabilistic reasoning in place, we can now assign symbols that represent e.g. objects in the scene since we can ground their unique attributes (e.g. location, embeddings) and track them. But probabilistic reasoning is not enough. In many problems we need to: 

1. Be able to accumulate knowledge (static rules and percepts) over a periods of time. 
2. Reason _beyond_ what we perceive.
3. Be able to express the domain and problem at hand in a suitably expressive language that can interface with domain-independent solvers to find solutions (usually via search algorithms). 

These are implemented in the planning subsystem that is our next logical step positioned after the probabilistic reasoning to provide the best sequence of actions to reach our goal.