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

The following python code is a self-contained implementation (requiring a plain text input file only) of the language model training above.  

{{<details "Minimal RNN language model code from Stanford CS231n" >}}

```python
# see here for notation http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture10.pdf
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file - you can use any (small) file in txt format from the web or type your own. 

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
  ```
  {{</details>}}

More concretely, to train an language model we need a big corpus of text which is a sequence of words $\mathbf x_1, ..., \mathbf x_{T}$ where T is the number of words / tokens in the corpus. Every time step we feed one word at a time to the RNN and and compute the output probability distribution $\mathbf \hat y_t$, which by construction is a _conditional_ probability distribution of every word in the dictionary given the words we have seen so far. The loss function at time step $t$ is the classic cross entropy loss between the predicted probability distribution and the distribution that corresponds to the one-hot encoded true next word. 

$$J_t(\theta) = CE(\hat  \mathbf y_t, \mathbf y_t) = - \sum_{w \in T} \mathbf y_t^{(w)} \log \hat \mathbf y_t^{(w)} = \log \hat \mathbf y_t^{(w)}$$ 

This is visually shown in the next figure for a hypothetical example of the shown sequence of words. 

![rnn-language-model-loss](images/rnn-language-model-loss.png#center)
*RNN Language Model Training Loss. For each input word (at step t$t$), the RNN predicts the next word and is penalized with a loss $J_t(\theta)$. The total loss is the average across the corpus.*

In practice we dont compute the total loss over the whole corpus but just like what we have done with DNN and CNN networks we train over a batch (sentence / document) and compute gradients over that span iterating on a stochastic gradient decent optimization algorithm. 
