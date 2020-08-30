---
title: Bayesian Inference 
weight: 82
draft: false
---

#  Bayesian Inference

## Bayes Theorem

The Bayesian theorem is the cornerstone of probabilistic modeling and ultimately governs what models we can construct inside the _learning algorithm_ of the previous section. If $\mathbf{w}$ denotes the unknown parameters, $\mathtt{data}$ denotes the dataset and $\mathcal{H}$ denotes the hypothesis set.

$$ p(\mathbf{w} | \mathtt{data}, \mathcal{H}) =  \frac{P(  \mathtt{data} | \mathbf{w}, \mathcal{H}) P(\mathbf{w} | \mathcal{H}) }{ P(  \mathtt{data} | \mathcal{H})} $$

The Bayesian framework allows the introduction of _priors_ from a wide variety of sources (experts, other data, past posteriors, etc.) For example,a medical patient is exhibiting symptoms x, y and z. There are a number of diseases that could be causing all of them, but only a single disease is present. A doctor (the expert) has beliefs about which disease, but a second doctor may have slightly different beliefs.

## Online Bayesian Regression

To see why lets consider an _online_ learning problem where the underlying target function is $p_{data}(x, \mathbf w) = w_0 + w_1 x + n$ This is the equation of a line. In this example its parametrized with $a_0=-0.3, a_1=0.5$ and $n \in \mathcal N(0, \sigma=0.2)$. To match the simple inference exercise that we just saw, we draw the equivalent PGM

<img src="images/Figure8.3.png" width="250" align="center">

*Bayesian Linear Regression example - please replace $t$ with $y$ to match earlier notation in these notes*
 
The Bayesian update of the posterior can be intuitively understood using a graphical example of our model of the form:
$$g(x,\mathbf{w})= w_0 + w_1 x$$ (our hypothesis). The reason why we pick this example is illustrative as the model has just two parameters and is amendable to visualization. The update needs a prior _distribution_ over $\mathbf w$ and a likelihood function. As prior we assume a spherical Gaussian 

$$p(\mathbf w | \alpha) = \mathcal N(\mathbf w | \mathbf 0, \alpha^{-1} \mathbf I)$$

with $\alpha = 0.2$. We starts in row 1 with this prior and at this point there are no data and the likelihood is undefined while every possible linear (line) hypothesis is feasible as represented by the red lines. In row 2, a data point arrives and the the Bayesian update takes place: the previous row posterior becomes the prior and is multiplied by the current likelihood function. The likelihood function and the form of the math behind the update are as shown in [Bishop's book in section 3.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf). Here we focus on a pictorial view of what is the update is all about and how the estimate of the posterior distribution $p(\mathbf w | \mathbf y)$ ultimately (as the iterations increase) it will be ideally centered to the ground truth ($\bm a$).

![Figure3.7-bishop](images/Figure3.7.png)
*Instructive example of Bayesian learning as data points are streamed into the learner. Notice the dramatic improvement in the posterior the moment the 2nd data point arrives. Why is that?*



