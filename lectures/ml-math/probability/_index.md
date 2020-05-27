---
title: Probability Basics
---

# Probability Basics

1. The whole purpose of probabilistic modeling is to introduce uncertainty into our problem statement. There are three types of uncertainties:

    * Inherent stochasticity - e.g. impact of wind in self-driving car control systems at moderate to high speed. 
    * Incomplete observability - e.g. sensor imperfections causing loss of sensing information
    * Incomplete modeling - e.g. models and algorithms that are not implementable to an analog world and need to be discretized.

2. Probabilities can be used in two ways.
    * Probabilities can describe frequencies of outcomes in random experiments
    * Probabilities can also be used, more generally, to describe degrees of belief in propositions that do not involve random variables. This more general use of probability to quantify beliefs is known as the Bayesian viewpoint. It is also known as the subjective interpretation of probability, since the probabilities depend on assumptions.

The three important probabilities that we need to be concerned with in data mining are the joint, marginal and conditional probabilities. The pictures below present some examples that we will go through.

![discrete-prob](images/discrete-prob.png#center)
*Discrete probability distribution function (pdf). On the right side is what is called a Hinton diagram where the area of the square represents the probability in a way that the sum equals to 1.0.*

![monogram](images/figure21.png#center)
*A more complicated discrete probability distribution over the letters of the English alphabet (letter 27 symbolizes space) as measured by reading the Linux FAQ document.*

![continuous-prob](images/continuous-prob.png#center)
*Continuous probability distribution density function (pdf) - the area under the curve equals  1.0*


![joint-bigram](images/figure2.2.png#center)
*Joint probability $P(x,y)$ distribution over the 27x27 possible bigrams $xy$ found in this document: https://www.tldp.org/FAQ/pdf/Linux-FAQ.pdf*

![joint-prob](images/joint-prob.png)
*(e) and (f) represent joint distributions of discrete and continuous random variables*


### Sum rule and the Marginal Probability

Given the joint what is the marginal probability $P(x)$ ?

$$P(x)   = \sum_y P(x,y) = \sum_y P(x | y)P(y)$$

![marginal-prob](images/marginal-prob.png#center)
*Extracting the marginal out of the joint*

### Conditional Probability and the Product or chain rule

This is obtained from the definition of conditional probability:

$$P(x,y) = P(x | y)P(y) = P(y | x)P(x)$$

![conditional-bigram](images/conditional-bigrams.png#center)
*Conditional probability distribution over the 27x27 possible bigrams $xy$ found in this document: https://www.tldp.org/FAQ/pdf/Linux-FAQ.pdf*

{{< hint warning >}}

Are $x$ and $y$ in the example above independent ?

{{< /hint >}}

## Probability Rules

These rules are the equivalent of the primitive calculus operations in the probability space. 




## Bayes Theorem

The Bayesian theorem is the cornerstone of probabilistic modeling. If $\mathbf{\theta}$ denotes the unknown parameters, $D$ denotes the dataset and $\mathcal{H}$ denotes the hypothesis space  - the model we have seen in [the learning problem]({{<ref "../../learning-problem">}}) chapter.

$$ P(\mathbf{\theta} | D, \mathcal{H}) =  \frac{P( D | \mathbf{\theta}, \mathcal{H}) P(\mathbf{\theta} | \mathcal{H}) }{ P(D|\mathcal{H})} $$

The Bayesian framework allows the introduction of priors from a wide variety of sources (experts, other data, past posteriors, etc.) For example,a medical patient is exhibiting symptoms x, y and z. There are a number of diseases that could be causing all of them, but only a single disease is present. A doctor (the expert) has beliefs about which disease, but a second doctor may have slightly different beliefs.

> NOTE: The [Probabilistic Programming & Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) book is one of the best resources out there containing practical python examples. In addition they have been recoded recently to work in [Tensorflow Probability](https://medium.com/tensorflow/an-introduction-to-probabilistic-programming-now-available-in-tensorflow-probability-6dcc003ca29e) an industrial-strength framework that can bring together Deep Learning and domain-specific probabilistic modeling. The book cant match the rigorousness of Bishop's book but it offers a good treatment on problems and use cases and should be considered complimentary.

## Multi-variate Gaussian distribution

Perhaps the only distribution that is worth knowing and remembering its form is the Multi-variate Normal as it has widespread applicability in data science. 

$$f_{\mathbf X}(x_1,\ldots,x_k) = \frac{\exp\left(-\frac 1 2 ({\mathbf x}-{\boldsymbol\mu})^\mathrm{T}{\boldsymbol\Sigma}^{-1}({\mathbf x}-{\boldsymbol\mu})\right)}{\sqrt{(2\pi)^n|\boldsymbol\Sigma|}}$$
where where <${\mathbf x}$ is a real 'n'-dimensional column vector and $|\boldsymbol\Sigma|\equiv \operatorname{det}\boldsymbol\Sigma$ is the determinant of $\boldsymbol\Sigma$. 

Apart from the definition, you need to connect the geometric interpretation of the bivariate Gaussian distribution to the eigendecomposition in the linear algebra lecture as shown below:

![bivariate-Gaussian](images/Figure2.7.png)

Such geometric interpretations will be very useful when we study dimensionality reduction via Principal Component Analysis (PCA).

## Book Chapters
 From Ian Goodfellow's book:
<iframe src="https://www.deeplearningbook.org/contents/prob.html" width="800" height="1200"></iframe>

We will go through the main points during the lecture and treat also [MacKay's book (Chapter 2)](https://www.inference.org.uk/itprnn/book.pdf) that is also instructive and a much better in introducing probability concepts. If you are a visual learner, [the visual information theory](http://colah.github.io/posts/2015-09-Visual-Information/#fn4) blog post is also a good starting point. 