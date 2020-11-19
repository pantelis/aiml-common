---
title: Neural Language Models
draft: false
weight: 134
---

# Neural Language Models

> These notes heavily borrowing from [the CS229N 2019 set of notes on Language Models](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes05-LM_RNN.pdf). 

Language modeling is the task of predicting (aka assigning a probability) what word comes next. More formally, given a sequence of words $\mathbf x_1, ..., \mathbf x_t$ the language model returns

$$p(\mathbf x_{t+1} | \mathbf x_1, ..., \mathbf x_t)$$

![language-model-google-search](images/language-model-google-search.png#center)
*Language Model Example*

How we can build language models though. One approach is to slide a window around the context we are interested in. 

![dnn-language-model](images/dnn-language-model.png#center)
*DNN language model - fixed sliding window around the context. The embeddings of each word (e.g. word2vec vectors) are represented by the blue layer and are being transformed via the weight matrix $\mathbf W$ to a hidden layer and from there via another transformation to a probability distribution. How do we determine the sliding window size? How to deal with the size of $\mathbf W$?*

To avoid the issues associated with the DNN, we will use the [RNN]({{<ref "../../rnn">}}) architectures we have seen in another chapter.  This is shown next for a toy example where the vocabulary is ['h','e','l','o']. where the tokens are single letters represented in the input with a one-hot encoded vector. 

![rnn-language-model](images/rnn-language-model.png#center)
*RNN language model example - training [ref](https://www.youtube.com/watch?v=6niqTuYFZLQ&t=521s). Note that in practice in the place of the on-hot encoded word vectors we will have word embeddings.*

 Let us assume that the network is being trained with the sequence "hello". The letters will come in one at a time, each letter going through the forward pass that produces at the output the $\mathbf y_t$ that indicates which letter is expected to arrive next.  You can see, since we are just started training,  that this network is not predicting correctly - this will improve over time as the model is trained with more sequence permutations form our limited vocabulary. During inference we will use the language model to generate the next token. 

![rnn-language-model-inference](images/rnn-language-model-inference.png#center)
*RNN language model example - generate the next token [ref](https://www.youtube.com/watch?v=6niqTuYFZLQ&t=521s)*

More concretely, to train an language model we need a big corpus of text which is a sequence of tokens $\mathbf x_1, ..., \mathbf x_{T}$ where T is the number of words / tokens in the corpus. 

Every time step we feed one word at a time to the RNN and compute the output probability distribution $\hat \mathbf y_t$, which by construction is a _conditional_ probability distribution of every word in the dictionary given the words we have seen so far. The loss function at time step $t$ is the classic cross entropy loss between the predicted probability distribution and the distribution that corresponds to the one-hot encoded true next token. 

$$J_t(\theta) = CE(\hat  \mathbf y_t, \mathbf y_t) = - \sum_{w \in T} \mathbf y_t^{(w)} \log \hat \mathbf y_t^{(w)} = \log \hat \mathbf y_t^{(w)}$$ 

This is visually shown in the next figure for a hypothetical example of the shown sequence of words. 

![rnn-language-model-loss](images/rnn-language-model-loss.png#center)
*RNN Language Model Training Loss. For each input word (at step t$t$), the RNN predicts the next word and is penalized with a loss $J_t(\theta)$. The total loss is the average across the corpus.*

In practice we don't compute the total loss over the whole corpus but just like what we have done with DNN and CNN networks we train over a finite span and compute gradients over that span iterating on a stochastic gradient decent optimization algorithm. Character-level language models have achieved [state of the art NLP results by Facebook Research](https://github.com/flairNLP/flair). 
