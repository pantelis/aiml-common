---
title: BERT 
draft: true
---
# BERT 
## The concept of Attention

When you hear the sentence "the soccer ball is on the field," you don’t assign the same importance to all 7 words. You primarily take note of the words "_ball_" "_on_," and "_field_" since those are the words that are most "important" to you.  

Using the final RNN hidden state as the single "context vector" for sequence-to-sequence models cant differentiate such significance. Moreover, different parts of the output
may even consider different parts of the input "important." For example, in translation, the first word of output is usually based on the first few words of the input, but the last word is likely based on the last few words of input.

Attention mechanisms make use of this observation by providing the decoder network with a look at the entire input sequence at every decoding step; the decoder can then decide what input words are important at any point in time. There are many types of attention
mechanisms - we focus here the [archetypical method](https://arxiv.org/abs/1409.0473). 

![seq2seq-attention](images/seq2seq-attention-step1.png#center)
*Attention in seq2seq neural machine translation - time step 1*

![seq2seq-attention](images/seq2seq-attention-step5.png#center)
*Attention in seq2seq neural machine translation - time step 5*

To implement the attention mechanism we need additional capabilities as follows:

1. During encoding the output of bidirectional LSTM encoder capture the contextual representation of each word via the encoder hidden vectors $h_1, ..., h_{Tx}$ where $Tx$ is the length of the input sentence. 

2. During decoding we compute the decoder hidden states using a recursive relationship

$$s_t = f (s_{t-1}, y_{t-1}, \phi_t)$$

Mathematically, our new model that incorporates attention maximizes a new conditional probability that now has time dependency in the context vector. 

$$p(\mathbf y | \mathbf x) = g(y_{t-1}, s_t, \phi_t)$$

For each hidden vector from the original sentence $h_j$ we compute a score

$$e_{t,j} = a(s_{t−1}, h_j)$$

where $a$ is any function with values in $\mathbb R$ for instance a single
layer fully-connected neural network. The score values are normalized 
using a softmax layer to produce the attention vector $\mathbf α_t$. 

The context vector $\phi_i$ is then the attention weighted average of the
hidden vectors from the original sentence. Intuitively, this vector captures the relevant contextual information from the original sentence for the t-th step of the decoder.

