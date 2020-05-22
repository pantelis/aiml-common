---
title: Neural Machine Translation
draft: false
weight: 135
---

# Neural Machine Translation

> These notes heavily borrowing from [the CS229N 2019 set of notes on NMT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf). 

![rosetta-stone](images/rosetta-stone.jpg#center)
*Rosetta Stone at the British Museum - depicts the same text in Ancient Egyptian, Demotic and Ancient Greek.*

Up to now we have seen how to generate [embeddings]({{<ref "../word2vec">}}) and predict a single output e.g. [the single most likely next word]({{<ref "../language-models">}}) in a sentence given the past few. However, there’s a whole class of NLP tasks that rely on sequential output, or outputs that are sequences of potentially varying length. For example,

* **Translation**: taking a sentence in one language as input and outputting the same sentence in another language.
* **Conversation**: taking a statement or question as input and responding to it.
* **Summarization**: taking a large body of text as input and outputting a summary of it.
* **Code Generation**: Natural Language to formal language code (e.g. python)

![ios-keyboard-translation](images/ios-keyboard-translation.png#center)
*Your smartphones translate the text you type in messaging applications to emojis and other symbols structured in such a way to convey the same meaning as the text". 

## Sequence-to-Sequence (Seq2Seq)  

Sequence-to-sequence, or "Seq2Seq", is a relatively new paradigm, with its [first published usage in 2014](https://arxiv.org/abs/1409.3215) for English-French translation. At a high level, a sequence-to-sequence model is an end-to-end model made up of two recurrent neural networks (LSTMs):

* an encoder, which takes the model’s input sequence as input and encodes it into a fixed-size "context vector" $\phi$, and

* a decoder, which uses the context vector from above as a "seed" from which to generate an output sequence.

For this reason, Seq2Seq models are often referred to as encoder-decoder models as shown in the next figure. 

![encoder-decoder-high-level](images/encoder-decoder-high-level.png#center)
*Encoder-Decoder NMT Architecture [ref](https://www.amazon.com/Natural-Language-Processing-PyTorch-Applications/dp/1491978236)*

The se2seq model is an example of _conditional language model_ because it conditions on the source sentence or its context $\phi$. It directly calculates,

$$p(\mathbf y| \mathbf x) = p(y_1| \mathbf x) p(y_2|y_1, \mathbf x ) ... p(y_T | y_1, ..., y_{T-1}, \mathbf x)$$

$$p(\mathbf y| \mathbf \phi) = p(y_1| \mathbf \phi) p(y_2|y_1, \mathbf \phi ) ... p(y_T | y_1, ..., y_{T-1}, \mathbf \phi)$$

### Encoder

The encoder network’s job is to read the input sequence to our Seq2Seq model and generate a fixed-dimensional context vector $\phi$ for the sequence. To do so, the encoder will use an LSTM – to read the input tokens one at a time. The final hidden state of the cell will then become $\phi$. However, because it’s so difficult to compress an arbitrary-length sequence into a single fixed-size vector (especially for difficult tasks like translation), the encoder will usually consist of stacked LSTMs: a series of LSTM "layers" where each layer’s outputs are the input sequence to the next layer. The final layer’s LSTM hidden state will be used as $\phi$.

![forward-backward-concat](images/forward-backward-concat.png#center)
*Bidirectional RNNs used for representing each word in the context of the sentence*

Mathematically the RNN evolves its hidden state as we have seen as,

$$h_t = f(x_t, h_{t-1})$$

and the context vector $\phi = q(h_1, ..., h_{Tx})$ is generated in general from the sequence of hidden states.  $f$ can be in e.g. any non-linear function such as an bidirectional LSTM with a given depth. 

Seq2Seq encoders will often do something strange: they will process the input sequence in reverse. This is actually done on purpose. The idea is that, by doing this, the last thing that the encoder sees will (roughly) corresponds to the first thing that the model outputs; this makes it easier for the decoder to "get started" on the output, which then gives the decoder an easier time generating a proper output sentence. In the context of translation, we’re allowing the network to translate the first few words of the input as soon as it sees them; once it has the first few words translated correctly, it’s much easier to go on to construct a correct sentence than it is to do so from scratch. 

In terms of architecture we usually have a deep (vertical direction) LSTM network whose unrolled view is shown next.

![lstm-nmt-encoder](images/lstm-nmt-encoder.png#center)
*Stacked LSTM Encoder (unrolled and showing the reverse direction only)*

### Decoder

The decoder is also an LSTM network, but its usage is a little more complex than the encoder network. Essentially, we’d like to use it as a language model that’s "aware" of the target words that it’s generated so far and of the input. To that end, we’ll keep the "stacked" LSTM architecture from the encoder, but we’ll initialize the hidden state of our first layer with the context vector from above; the decoder will literally use the context of the input to generate an output.

Once the decoder is set up with its context, we’ll pass in a special token to signify the start of output generation; in literature, this is usually an <EOS> token appended to the end of the input (there’s also one at the end of the output). Then, we’ll run all three layers of LSTM, one after the other, following up with a softmax on the final layer’s output to generate the first output word. Then, we pass that word into the first layer, and repeat the generation. This is a technique called **Teacher Forcing** wherein the input at each time step is given as the actual output (and not the predicted output) from the previous time step.  

![lstm-nmt-decoder](images/lstm-nmt-decoder.png#center)
*LSTM Decoder (unrolled). The decoder is a language model that’s "aware" of the words that it’s generated so far and of the input.*

Once we have the output sequence, we use the same learning strategy as usual. We define a loss, the cross entropy on the prediction sequence, and we minimize it with a gradient descent algorithm and back-propagation. _Both_ the encoder and decoder are trained at the same time, so that they both learn the same context vector representation as shown next. 

![seq2seq-training](images/seq2seq-training.png#center)
*Seq2Seq Training - backpropagation is end to end.*

{{<details "Python Code for NMT">}}

You need to go through [this](https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7) implementation to understand the basic NMT mechanism and the dimensioning of the RNN (LSTMs) involved. 

{{</details>}}

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