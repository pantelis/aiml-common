---
title: Word Embeddings
draft: false
weight: 132
---

# Word Embeddings 

In the so called classical NLP, words were treated as atomic symbols, e.g. `hotel`, `conference`, `walk` and they were represented with on-hot encoded (sparse) vectors e.g.

$$\mathtt{hotel} = [0,0, ..., 0,1, 0,0,..,0]$$
$$\mathtt{motel} = [0,0, ..., 0,0, 0,1,..,0]$$

The size of the vectors is equal to the vocabulary size $V$. These vocabularies are very long - for speech recognition we may be looking at 20K entries but for information retrieval on the web google has released vocabularies with 13 million words (1TB). In addition such representations are orthogonal to each other by construction - there is no way we can relate `motel` and `hotel` as their dot product is 

$$ s = \mathtt{hotel}^T \mathtt{motel} = 0.0$$

One of key ideas that made NLP successful is the _distributional semantics_ that originated from Firth's work: _a word’s meaning is given by the words that frequently appear close-by_. When a word $x$ appears in a text, its context is the set of words that appear nearby (within a fixed-size window). Use the many contexts of $x$ to build up a representation of $x$. 

![distributional-similarity](images/distributional-similarity.png#center)
*Distributional similarity representations - `banking` is represented by the words left and right across all sentences of our corpus.*

This is the main idea behind word2vec word embeddings (representations) that we address next. 

## Word2Vec

In 2012, Thomas Mikolov, an _intern_ at Microsoft, [found a way](https://arxiv.org/abs/1310.4546)[^2] to encode the meaning of words in a modest number of vector dimensions $d$. Mikolov trained a neural network to predict word occurrences near each target word. In 2013, once at Google, Mikolov and his teammates released the software for creating these word vectors and called it word2vec. 

![banking-vector](images/banking-vector.png#center)
*word2vec generated embedding for the word `banking` in d=8 dimensions*

Here is a [visualization](http://projector.tensorflow.org/) of these embeddings in the re-projected 3D space (from $d$ to 3). Try searching for the word "truck" for the visualizer to show the _distorted_ neighbors of this work - distorted because of the 3D re-projection. In another example, word2vec embeddings of US cities projected in the 2D space result in poor topological but excellent _semantic_ mapping which is exactly what we are after. 

![semantic-map-word2vec](images/semantic-map-word2vec.png#center)
*Semantic Map produced by word2vec for US cities*

Another classic example that shows the power of word2vec representations to encode analogies, is  classical king + woman − man ≈ queen example shown below.

![queen-example](images/queen-example.png#center)
*Classic queen example where `king − man ≈ queen − woman`, and we can visually see that in the red arrows. There are 4 analogies one can construct, based on the parallel red arrows and their direction. This is slightly idealized; the vectors need not be so similar to be the most similar from all word vectors. The similar direction of the red arrows indicates similar relational meaning.*

So what is the more formal description of the word2vec algorithm? We will focus on one of the two computational algorithms[^1] - the skip-gram method and use the following diagrams as examples to explain how it works. 

![word2vec-idea](images/word2vec-idea.png#center)
*Computing $p(w_{t+j}|w_t)$ with word2vec*

![word2vec-idea2](images/word2vec-idea2.png#center)
*Computing $p(w_{t+j}|w_t)$ with word2vec*

The algorithm starts with all word embeddings in our corpus being random vectors. 

It then iteratively goes through each position $t$ in each sentence and for the center word at that location $w_t$ we predict the outside words $w_{t+j}$ where $j$ is over a window of size $C = |\\{ j: -m \le j \le m \\}|-1$ around $w_t$. In other words we need to calculate the probability $p(w_{t+j}|w_t)$ and in each iteration we adjust the word vectors to maximize this probability. 

So for example, the meaning of `banking` is predicting the context (the words around it) in which `banking` occurs across our corpus.  The term _prediction_ remind us the discussion we had in the [linear regression]({{<ref "../../regression">}}) section and the maximum likelihood principle. Therefore the objective is to minimize the negative log likelihood that is,

$$J(\theta) = -\frac{1}{V} \log L(\theta) = -\frac{1}{V} \sum_{t=1}^V \sum_{-m\le j \le m, j \neq 0} \log p(w_{t+j} | w_t; \theta)$$

where $V$ is the size of the vocabulary (words) that results after executing, in a corpora of documents, some of the steps we have seen in the [introductory]({{<ref "../nlp-intro">}}) section. T could be very large - 1 billion words. The ML principle, powered by a corresponding algorithm, will result into a model that for each word at the input will predict the context words around it.  The model parameters $\theta$ will need to be optimized based on a training dataset that we soon see how we create. The parameters $\theta$ are just the vector representations of each word in the training dataset. 

So the question now becomes how to calculate $p(w_{t+j} | w_t; \theta)$ and we do so with the network architecture below. 

![word2vec-network](images/word2vec-network.png#center)
*Conceptual architecture of the neural network that learns word2vec embeddings. The text refers to the hidden layer dimensions as $d$ rather than $N$ and hidden layer $\mathbf z$ rather than $\mathbf h$.*

The network accepts the center word and via an embedding layer $\mathbf W_{V \times d}$ produces a hidden layer $\mathbf z$. The same hidden layer output is then mapped to an output layer of size $C \times V$, via $\mathbf W^\prime_{d \times V}$. One mapping is done for each of the words that we include in the context. In the output layer we then convert the metrics $\mathbf z^\prime$ to a probability distribution $\hat{\mathbf y}$ via the softmax. This is summarized next:

$$\mathbf z = \mathbf x^T \mathbf W$$
$$\mathbf z^\prime_j = \mathbf z \mathbf W^\prime_j,  j \in \[1,...,C]$$
$$\hat{\mathbf y}_j = \mathtt{softmax}(\mathbf z^\prime), j \in \[1,...,C]$$
$$L = CE(\mathbf{y}, \hat{\mathbf y} )$$

The parameters $\theta = \[ \mathbf W, \mathbf W^\prime \]$ will be optimized via an optimization algorithm (from the usual SGD family). The only missing piece now is the training data. Training data can be easily generated from our corpus assuming we fix the $m$. Lets see an example:

![training-data-word2vec](images/training-data-word2vec.png#center)
*Training data generation for the sentence 'Claude Monet painted the Grand Canal of Venice in 1908'*

Training for large vocabularies can be quite computationally intensive. This [self-contained implementation](https://github.com/ujhuyz0110/wrd_emb/blob/master/word2vec_skipgram_medium_v1.ipynb) is instructive and you should [go through it](https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72) to understand the algorithm analysis above. Exercising the `gensim` and other library APIs is easy especially with pre-trained models, but to really understand a topic, you need to code something from scratch. 

At the end of training we are then able to store the matrix $\mathbf W$ and load it during the parsing stage of the NLP pipeline.  

[^1]: The other method is called Continuous Bag of Words (CBOW) and its the reverse of the skip-gram method: it predicts the center word from the words around it. Skip-gram works well with small corpora and rare terms while CBOW shows higher accuracies for frequent words and is faster to train [ref](https://www.manning.com/books/natural-language-processing-in-action). 

[^2]: You don't come across papers with 10K citations very often. 

As quoted by Noah Smith [cite](https://arxiv.org/abs/1902.06006), the notable disadvantage of reduced-dimensionality vectors is that the individual dimensions are no longer interpretable features that can be mapped back to intuitive building blocks contributing to the word’s meaning.   The word’s meaning is distributed across the whole vector;  for this reason,  these vectors are sometimes called _distributed representations_.
