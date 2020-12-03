---
title: Attention
---

# Attention

When you hear the sentence "the soccer ball is on the field," you don’t assign the same importance to all 7 words. You primarily take note of the words "_ball_" "_on_," and "_field_" since those are the words that are most "important" to you.  

Using the _final_ RNN hidden state as _the_ single "context vector" for sequence-to-sequence models cant differentiate such significance. Moreover, different parts of the output may even consider different parts of the input "important." For example, in translation, the first word of output is usually based on the first few words of the input, but the last word is likely based on the last few words of input. For example, "Tech giants" and "Les géants de la technologie".

Attention mechanisms make use of this observation by providing the decoder network with a look at the entire input sequence at every decoding step; the decoder can then decide what input words are important at any point in time. There are many types of attention
mechanisms - we focus here the [archetypical method](https://arxiv.org/abs/1409.0473). 

To implement the attention mechanism we need additional capabilities:

1. During encoding the output of bidirectional LSTM encoder can provide the contextual representation of _each word_ via the encoder hidden vectors $h_1, ..., h_{Tx}$ where $Tx$ is the length of the input sentence. 

2. During decoding we compute the RNN decoder hidden states using a recursive relationship,

$$s_t = g(s_{t-1}, y_{t-1}, \phi_t)$$

where $s_{t-1}$ is the previous hidden vector, $y_{t-1}$ is the generated word at the previous step and $\phi_t$ is the context vector that that captures the context from the original sentence that is relevant to the decoder at time $t$.  

Mathematically, our new model that incorporates attention maximizes a new conditional probability that now has **time dependency** in the context vector. 

$$p(\mathbf y | \mathbf x) = g(y_{t-1}, s_t, \phi_t)$$

The $\phi_t$ is computed as follows: 

1. For each hidden vector from the original sentence $h_j$, $j$ is the index of the input sentence, we compute a score

$$e_{t,j} = a(s_{t−1}, h_j)$$

where $a$ is any function with values in $\mathbb R$ for instance a single layer fully-connected neural network. 

2. The score values are normalized  using a softmax layer to produce the attention vector $\mathbf α_t$. 

$$\mathbf a_t = softmax(\mathbf e_{t})$$

3. The context vector $\phi_t$ is then the attention weighted average (dot product) of the hidden vectors from the original sentence. 

$$ \phi_t = \sum_j \alpha_{t,j}h_j$$

Intuitively, this vector captures the relevant contextual information from the original sentence for the t-th step of the decoder.

<!-- 
![seq2seq-attention](images/seq2seq-attention-step1.png#center)
*Attention in seq2seq neural machine translation - time step 1*

![seq2seq-attention](images/seq2seq-attention-step5.png#center)
*Attention in seq2seq neural machine translation - time step 6* -->
