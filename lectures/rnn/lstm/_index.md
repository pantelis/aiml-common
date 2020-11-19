---
title: The Long Short-Term Memory (LSTM) Cell Architecture
draft: false
weight: 135
---

# The Long Short-Term Memory (LSTM) Cell Architecture

In [the simple RNN]({{<ref "../simple-rnn/">}}) we have seen the problem of exploding or vanishing gradients when [the span of back-propagation is large](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf) (large $\tau$). Using the conceptual IIR filter, that ultimately _integrates_ the input signal, we have seen that in order to avoid an exploding or vanishing impulse response, we need to control $w$. This is exactly what is being done in evolutionary RNN architectures that we will treat in this section called _gated RNNs_. The best known gated RNN architecture is called the LSTM _cell_ and in this case the weight $w$ is not fixed but it is determined based on the input sequence _context_. The architecture is shown below. 

![lstm-cell](images/rnn-LSTM.png#center)
*LSTM architecture: It is divided into three areas: input (green), cell state (blue) and output (red). You can clearly see the outer ($\bm h_{t-1}$ )and the inner ($\bm s_{t-1}$) recurrence loops.*

Because we need to capture the input context that involve going back several time steps in the past, we introduce an _additional_ inner recurrence loop that is effectively a variable length internal to the cell memory - we call this the _cell state_.  We employ another hidden unit called the _forget gate_  to learn the input context and the forgetting factor (equivalent to the $w$ we have seen in the IIR filter) i.e. the extent that the cell forgets the previous hidden state. We employ a couple of other gates as well: the _input gate_ and the _output gate_ as shown in the diagram below. In the following we are describing what each component is doing. 

## The Cell State

Starting at the heart of the LSTM cell, to describe the update we will use two indices: one for the unfolding sequence index $t$ and the other for the cell index $i$. We use the additional index to allow the current cell at step $t$ to use or forget inputs and hidden states from other cells. 

$$s_t(i) = f_t(i) s_{t-1}(i) + g_t(i) \sigma \Big( \bm W^T(i) \bm h_{t-1}(i) + \bm U^T(i) \bm x_t(i) + \bm b(i) \Big)$$

The parameters $\theta_{in} = \\{  \bm W, \bm U, \bm b \\}$  are the recurrent weights, input weights and bias at the input of the LSTM cell. Please note that in the above equation some authors use a $\tanh$ non-linearity to transform the input instead of sigmoid.
 
The forget gate calculates the forgetting factor,

$$f_t(i) =\sigma \Big( \bm W_f^T(i) \bm h_{t-1}(i) + \bm U_f^T(i) \bm x_t(i) + \bm b_f(i) \Big) $$

## Input

The input gate _protects the cell state_ contents from perturbations by irrelevant to the context inputs. Quantitatively,  input gate calculates the factor,

$$g_t(i) =\sigma \Big( \bm W_g^T(i) \bm h_{t-1}(i) + \bm U_g^T(i) \bm x_t(i) + \bm b_g(i) \Big) $$

The gate with its sigmoid function adjusts the value of each element produced by the input neural network.

## Output

The output gate _protects the subsequent cells_ from perturbations by irrelevant to their context cell state. Quantitatively,

$$h_t(i) = q_t(i) \tanh(s_t(i))$$ 

where $q_t(i)$ is the output factor

$$q_t(i) =\sigma \Big( \bm W_o^T(i) \bm h_{t-1}(i) + \bm U_o^T(i) \bm x_t(i) + \bm b_o(i) \Big) $$

Notice that if you make the output of input and output gates equal to 1.0 and the forgetting factor equal to 0.0, we are back to the simple RNN architecture. You can expect backpropagation to work similarly in LSTM albeit with more complicated expressions. 


## Additional Resources

Additional tutorial resources on LSTMs can be found here:
1. [A Critical Review of Recurrent Neural Networks for Sequence Learning](https://arxiv.org/pdf/1506.00019.pdf)
2. [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs) 
3. [Illustrated guide to LSTMs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
4. [Simplest possible LSTM explanation video](https://www.youtube.com/watch?v=WCUNPb-5EYI)
