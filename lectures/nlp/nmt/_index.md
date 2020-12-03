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
*Your smartphones translate the text you type in messaging applications to emojis and other symbols structured in such a way to convey the same meaning as the text*. 

## Sequence-to-Sequence (Seq2Seq)  

Sequence-to-sequence, or "Seq2Seq", is a relatively new paradigm, with its [first published in 2014](https://arxiv.org/abs/1409.3215) that treated English-French translation. At a high level, a sequence-to-sequence model is an end-to-end model made up of two recurrent neural networks (LSTMs):

* an encoder, which takes the a source sequence as input and encodes it into a fixed-size "context vector" $\phi$, and

* a decoder, which uses the context vector from above as a "seed" from which to generate an output sequence.

For this reason, Seq2Seq models are often referred to as encoder-decoder models as shown in the next figure. 

![encoder-decoder-high-level](images/encoder-decoder-high-level.png#center)
*Encoder-Decoder NMT Architecture [ref](https://www.amazon.com/Natural-Language-Processing-PyTorch-Applications/dp/1491978236)*

### Encoder

The encoder network’s job is to read the input sequence to our Seq2Seq model and generate a fixed-dimensional context vector $\phi$ for the sequence. To do that we use an RNN (LSTM) that mathematically, it evolves its hidden state as we have seen as,

$$\mathbf h_t = f(\mathbf x_t, \mathbf h_{t-1})$$

and the context vector $\mathbf \phi = q(\mathbf h_1, ..., \mathbf h_{Tx})$ is generated in general from the sequence of hidden states.  $f$ can be in e.g. any non-linear function such as an _bidirectional_ LSTM with a given depth. The bidirectional RNN is shown schematically below. 


![forward-backward-concat](images/forward-backward-concat.png#center)
*Bidirectional RNNs used for representing each word in the context of the sentence*


In this architecture, we read the input tokens one at a time. The final hidden state of the cell will then become $\phi$. However, because it’s so difficult to compress an arbitrary-length sequence into a single fixed-size vector (especially for difficult tasks like translation), the encoder will usually consist of stacked LSTMs: a series of LSTM "layers" where each layer’s outputs are the input sequence to the next layer. The final layer’s LSTM hidden state will be used as $\phi$.

![lstm-nmt-encoder](images/lstm-nmt-encoder.png#center)
*Stacked LSTM Encoder (unrolled and showing the reverse direction only)*

In addition, Seq2Seq encoders will often do something strange: they will process the input sequence in reverse. This is actually done on purpose. The idea is that, by doing this, the last thing that the encoder sees will (roughly) corresponds to the first thing that the model outputs; this makes it easier for the decoder to "get started" on the output. 

### Decoder

The decoder is a language model that’s "aware" of the target words that it’s generated so far and of the input. In fact it is an example of _conditional language model_ because it conditions on the source sentence or its context $\phi$. The context $\phi$ can be calculated via

$$ \phi = q(\mathbf h_1, \mathbf h_2 \dots \mathbf h_{Tx})$$

NOTE: It is unfortunate but the Greek letters cant be boldfaced fonts on this site.

For example, in the simplest case, $\mathbf \phi = \mathbf h_{Tx}$

The decoder directly calculates,

$$\hat \mathbf y  = \argmax_y p(\mathbf y | \mathbf \phi)$$

We can write this as:

$$\argmax_y p(\mathbf y| \mathbf x) = p(y_1, y_2, ..., y_{Ty} | \phi)$$

and then use the product rule of probability to decompose this to:

$$\argmax_y p(y_{Ty} | y_1, ..., y_{T_y-1}, \mathbf \phi)  \dots p(y_2|y_1, \mathbf \phi ) p(y_1| \mathbf \phi) $$

We can now write,

$$\hat \mathbf y  = \argmax_y p(\mathbf y | \mathbf \phi) = \prod_{t=1}^{T_y} p(y_t | y_1, ..., y_{t-1}, \mathbf \phi) $$

In this equation $p(y_t | y_1, ..., y_{t-1}, \mathbf \phi)$ is a probability distribution represented by a softmax across all all the words of the dictionary. We can use an RNN (LSTM) to model the conditional probabilities 

$$LSTM = g(s_t,  y_{t-1}, \phi ) = p(y_t | y_1, ..., y_{t-1}, \mathbf \phi) $$

To that end, we’ll keep the "stacked" LSTM architecture from the encoder, but we’ll initialize the hidden state of our first layer with the context vector; the decoder will literally use the context of the input to generate an output. Once the decoder is set up with its context, we’ll pass in a special token to signify the start of output generation; in literature, this is usually an <EOS> token appended to the end of the input (there’s also one at the end of the output). Then, we’ll run all stacked layers of LSTM, one after the other, following up with a softmax on the final layer’s output to generate the first output word. Then, we pass that word into the first layer, and repeat the generation. This is a technique called **teacher forcing** wherein the input at each time step is given as the actual output (and not the predicted output) from the previous time step.  

![lstm-nmt-decoder](images/lstm-nmt-decoder.png#center)
*LSTM Decoder (unrolled). The decoder is a language model that’s "aware" of the words that it’s generated so far and of the input.*

Once we have the output sequence, we use the same learning strategy as usual. We define a loss, the cross entropy on the prediction sequence, and we minimize it with a gradient descent algorithm and back-propagation. _Both_ the encoder and decoder are trained at the same time, so that they both learn the same context vector representation as shown next. 

![seq2seq-training](images/seq2seq-training.png#center)
*Seq2Seq Training - backpropagation is end to end.*

## Metrics - BLEU

In 2002, IBM researchers developed the Bilingual Evaluation Understudy (BLEU) that remains, with its many variants to this day, one of
the most respected and reliable methods for machine translation.
### [Intuition](https://en.wikipedia.org/wiki/BLEU)

The BLEU algorithm evaluates the precision score of a candidate machine translation against a reference human translation. To refresh your memory the following picture is a easy picture for you to memorize and draw every time you forget the definitions of precision and recall.

![precision-recall](images/precision-recall.png#center)

The reference human translation is a assumed to be a model example of a translation, and we use _n-gram_ matches as our metric for how similar a candidate translation is to it. Why simple precision can't be used? Consider the following example of poor machine translation output with high precision metric using unigrams. 

| Output | | | | | |
| --- | --- | --- | --- | --- | --- |  --- |
| Candidate	| the |	the	| the | the	| the | the | the | 
| Reference 1 |	the | cat | is	| on | the | mat |	
| Reference 2 | there | is	| a	| cat	| on	| the	| mat |

Of the seven words in the candidate translation, all of them appear in the reference translations. Thus the candidate text is given a unigram precision of,

$$ P= \frac{m}{w_{t}} = \frac{7}{7}=1$$

where $m$ is number of words from the candidate that are found in the reference, and $w_t$ is the total number of words in the candidate. This is a perfect score, despite the fact that the candidate translation above retains little of the content of either of the references.

The modification that BLEU makes is fairly straightforward. For each word in the candidate translation, the algorithm takes its maximum total count, $m_{max}$, in any of the reference translations. In the example above, the word "the" appears twice in reference 1, and once in reference 2. Thus $m_{max}=2$.

For the candidate translation, the count $m_{w}$ of each word is clipped to a maximum of $m_{max}$ for that word. In this case, "the" has $m_w=7$ and $m_{max}=2$ thus $m_w$ is clipped to 2. These clipped counts $m_w$ are then summed over all distinct words in the candidate. This sum is then divided by the total number of unigrams in the candidate translation. In the above example, the modified unigram precision score would be:

$$P= \frac{2}{7}$$

### Calculation

A simple practical example is shown next. Note that the weights are specified as a tuple where each index refers to the gram order. To calculate the BLEU score only for 1-gram matches, you can specify a weight of 1 for 1-gram and 0 for 2, 3 and 4 (1, 0, 0, 0). 

```python
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)
```

In practice, however, using individual words as the unit of comparison is not optimal. Instead, BLEU computes the same modified precision metric using n-grams. The length which has the "highest correlation with monolingual human judgement" was found to be **four**. This is also the default in many packages that calculate BLUE scores (BLUE-4 metric).

```python
# n-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test']
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
```

Cumulative scores refer to the calculation of individual n-gram scores at all orders from 1 to n and weighting them by calculating the weighted _geometric mean_.


The BLEU score more formally consists of a brevity penalty $\beta$:

$$\beta = \exp(\min(0, 1- \frac{len(R)}{len(C)}))$$

where $len()$ is the length operator as applied in the reference (R) and candidate (C) sentences. 

$$BLEU = \beta \prod_{i=1}^k p_n^{w_n}$$

where $p_n$ is the n-gram precision and this makes BLUE a  _precision_ oriented metric. You can remember it easily if you remember to draw the diagram above and make the following associations

| Metric | Description as it applies to the NMT |
| --- | --- |
| TP | There are n-grams in C that are in R | 
| FP | There are n-grams in C that are not in R | 
| FN | There are missing n-grams in C that are in R | 

BLUE is concerned with the first two rows and it shouldn't be a surprise that the n-gram precision is ratio of the counts $\frac{TP}{TP+FP}$.



