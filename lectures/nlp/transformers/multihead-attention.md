# Multi-head attention 

For the input sentence 

"I usually go to the lab to run my experiments and to meet my colleagues."

we understand that multiple relationships must be understood with respect to the noun "lab". For example, we need to understand that "lab" is a place where experiments are run, but also a place where we meet our colleagues.

To capture such multiplicity of higher order meanings, we can use multiple attention heads. Each attention head will learn a different domain between the input tokens and will do so at the same time (in parallel).

![](images/multi-head-attention.png)

The complexity of running multiple heads does not scale with the number of heads since the number of parameters is shared across the heads.

## Resources

1. [Dimensioning Transformers - Part 1 ](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

2. [Dimensioning Transformers - Part 2](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)
