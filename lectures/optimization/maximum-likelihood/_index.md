---
title: Maximum Likelihood (ML) Estimation 
---

# Maximum Likelihood (ML) Estimation

Most of the models in supervised machine learning are estimated using the ML principle. In this section we introduce the principle and outline the objective function of the ML estimator that has wide applicability in many learning tasks. 

## Learning marginal models -  $p_{model}(\mathbf{x}; \mathbf w)$

Assume that we have $m$ examples drawn from a data generator that generates the vectors $\mathbf{x} \in \mathcal{X}$ _independently and identically distributed (i.i.d.)_ according to some unknown (but fixed) probability distribution function $p_{data}(\mathbf{x})$.

$$\mathbb{X} = \\{ \mathbf{x}_1, \dots, \mathbf{x}_m \\}$$

Let $p_{model}(\mathbf x, \mathbf w)$ a parametric family of probability distributions (our hypothesis set) over the same space that attempts to approximate (model) $p_{data}(\mathbf{x})$ as closely as possible using a suitable estimate of the parameter vector $\mathbf w$. The ML estimator for $\mathbf w$ is defined as:

$$\bm w_{ML} = \argmax_{\mathbf w} p_{model}(\mathbb X; \mathbf w)$$
$$ = \argmax_{\mathbf w} \prod_{i=1}^m p_{model}(\mathbf x^{(i)}; \mathbf w)$$
$$ = \argmax_{\mathbf w} \sum_{i=1}^m \log p_{model}(\mathbf x^{(i)}; \mathbf w)$$
$$ = \argmax_{\mathbf w} \frac{1}{m} \sum_{i=1}^m \log p_{model}(\mathbf x^{(i)}; \mathbf w)$$
$$ = \argmax_{\mathbf w} \mathbb{E}_{\mathbf{x} \sim \hat p_{data}} \log p_{model}(\mathbf x; \mathbf w)$$

From the last expression it is evident that in ML estimation two distributions are involved: $\hat p_{data}$ and $p_{model}$. The ML estimator minimizes the distance between the two empirical distributions therefore the KL divergence since.  

$$KL( \hat p_{data} || p_{model} ) = \mathbb{E}_{\mathbf x  \sim \hat p_{data}} \[\log \hat p_{data}(\mathbf x) - \log p_{model}(\mathbf x, \mathbf w) \]$$ 

The term on the left is independent of the model and therefore we only need to minimize the cost function:

$$ L(\mathbf w) = - \mathbb{E_{\mathbf x \sim \hat p_{data}}}  \log p_{model}(\mathbf x, \mathbf w)$$

Based on the last expression **minimizing the KL divergence, is equivalent in minimizing cross-entropy (CE)**. 

It is now instructive to go over an example to understand that even the plain-old mean squared error (MSE), the objective that is common in the regression setting, falls under the same umbrella - its the cross entropy between $\hat p_{data}$ and a Gaussian model. 


## Learning conditional models - $\hat p_{model}(\mathbf y | \mathbf x; \mathbf w)$

Cross entropy is in fact a very generic objective (loss) function that is applicable to any learning problem that uses maximum likelihood to estimate a model.  The discussion in the marginal distribution  is equivalently applicable to the conditional distribution $p_{model}(\mathbf y | \mathbf x, \mathbf w)$ which governs supervised learning, $y$ being the symbol of the label / target variable. Therefore all machine learning software frameworks offer excellent APIs on CE calculation.

$$ L(\mathbf w) = - \mathbb{E_{\mathbf x, \mathbf y \sim \hat p_{data}}}  \log p_{model}(\mathbf y | \mathbf x; \mathbf w)$$

The attractiveness of the ML solution is that the CE (also known as log-loss) is general and we _don't need to re-design it_ when we change the model.   

### Example: 

Please follow the whiteboard discussion associated with Section 5.5.1 and consider the following figure for assistance to visualize what is going on.

![conditional-model-gaussian](images/conditional-model-gaussian.png#center)
