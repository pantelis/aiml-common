---
title: Generative Modeling and Continuous Variational Auto Encoders (VAE)
draft: true
weight: 220
---

# Generative Modeling and  Continuous Variational Auto Encoders (VAE)

We have seen in the treatment of [CNNs]({{<ref "../../cnn/cnn-intro" >}}) that they can generate features that are suitable for the classification or regression task at hand using the labels to guide the maximization of the log-likelihod function. Here we are looking at the problem where we need features that are suitable for generating data from the input distribution without necessarily having labels. In this setting we will look deeper into a major family of variational inference: the VAE. Variational Autoencoders (VAEs) are popular generative models being used in many different domains, including collaborative filtering, image compression, reinforcement learning, and generation of music and sketches.

## Generative Modeling and Approximate Inference

In generative modeling we want to model the generative distribution of the observed variables $\mathbf x$, $p(\mathbf x)$ [^1]. 

[^1]: Following [this tutorial](https://arxiv.org/pdf/1906.02691.pdf) we adopt the compact notation that serializes into one vector $\mathbf x$ all the observed random variables. 

Take for example a set of images that depict a pendulum excited by a force. We would like to generative the successive frames of the pendulum swinging behavior and we do so we are assisted by a set of latent variables $\mathbf z$ that represent the underlying laws of physics. Generative modeling is especially well suited for 

1. Testing out hypotheses about the underlying rules that generated the observed data. Such rules can also offer interpretable models. 
2. Ability to capture causal relationships, since the ability of a factor to generate data very close to the ones observed offers a strong indication of such relationship.
3. Semi-supervised classification where the generated data are very close to already labeled data and therefore can improve classification model accuracy. 

One of the main methods of generative approximate inference is _variational inference_ and VAE is a modified instantiation of such inference and it involves: (a) deep latent variable models and (b) inference models both learned using stochastic gradient descent. Before we go deeper into what VAE is, its worth motivating the discussion as to why it came to be the solution to the generative problem we face.   

In probabilistic modeling we usually make use of latent variables $\mathbf z$, variables that are not observed but can be used to build suitable representational constraints in our models, and a set of parameters $\theta$ that parametrize the latent variable model $p(\mathbf x, \mathbf z | \mathbf \theta)$. Since, 

$$p(\mathbf x | \mathbf \theta) =  \sum_{\mathbf z} p(\mathbf x, \mathbf z | \mathbf \theta) $$

to generate new data whose marginal is ideally identical to the true but unknown target distribution we need to be able to sample from $p(\mathbf x, \mathbf z | \mathbf \theta)$.

The introduction of the latent variables can be represented as directed graph abd we have seen in the [probabilistic graphical models]({{<ref "../../pgm/pgm-intro">}}) introduction, the representation as directed graph allows the factorization of the joint distribution 

$$p(\mathbf x_1, \mathbf x_2, \dots, \mathbf x_M | \mathbf \theta) = \prod_{j=1}^M p(\mathbf x_j | Pa(\mathbf x_j))$$

where $Pa()$ is the set of parent nodes that the variable $\mathbf x_j$ is dependent on (has directed edges with). 

Consider our simple PGM, shown below: 

![vae-pgm](images/vae-pgm.png#center)
*Probabilistic Graphical Model from [here](https://arxiv.org/pdf/1606.05908.pdf)*

To generate from the marginal we need to implement a generative model that is a direct consequence of the chain and total probability rules. 

$$ p(\mathbf x | \mathbf \theta)  = \sum_{\mathbf z} p(\mathbf x| \mathbf z ; \mathbf \theta) p(\mathbf z | \theta)$$ 

The elements of this model are $p(\mathbf x| \mathbf z ; \mathbf \theta)$ and the $p(\mathbf z | \theta)$ that is often called the _prior distribution_ over $\mathbf z$. One of the methods of generating such samples is to start from a very easy to sample distribution and use function approximation that maps _variables_ to the distribution _parameters_ over these variables. One of the best function approximators that scale very well for the, usually large, dataset sizes we face are Deep Neural Networks (DNNs). When DNNs are used, we say that we implement a  _deep latent variable model_ that involves the following two steps: 

$$ \mathbf \eta = f_{DNN}(Pa(\mathbf x))$$
$$p(\mathbf x_j | Pa(\mathbf x_j), \mathbf \theta) = p(\mathbf x | \mathbf \eta, \mathbf \theta)$$

In deep latent variable models of the form we are concerned with, we select an easy to sample from, prior distribution

$$ p(\mathbf z) = Normal(0, \sigma^2 I)$$

and let the DNN implement the mapping

$$ \mathbf \eta = f_{DNN}(\mathbf z)$$
$$p(\mathbf x, \mathbf \theta) = p(\mathbf x | \mathbf \eta, \mathbf \theta)$$

However we are facing the following situation: _Even with DNNs aka when we let the DNN "design" the right feature coordinates in the latent space [^2], we are still facing an intractable computationally model in trying to estimate the marginal distribution $p(\mathbf x | \mathbf \theta)$_.  

[^2]: Note that the features that the DNN captures are not interpretable as the intuitively understood features that humans consider. For the MNIST dataset for example, humans will consider the slant of each digit, thinner strokes etc.

To understand why, consider the MNIST dataset and the problem of generating handwritten digits that look like that. We can sample from $p(\mathbf z | \theta)$ generating a large number of samples $\{z_1, \dots , z_k}$, since the DNN provided all the parameters of this distribution.  We can then compute $p(\mathbf x) = \frac{1}{k} \sum_i p(\mathbf x|z_i)$. The problem is that we need a very large number of such samples in high dimensional spaces such as images (for MNIST is 28x28 dimensions) . Most of the samples $\mathbf z_i$ will result into negligible $p(\mathbf x|z_i)$ and therefore won't contribute to the estimate of the $p(\mathbf x)$. This is the problem that VAE addresses. The key idea behind its design is that of _inference_ of the right latent space that when sampled, results into a computation and optimization of the marginal distribution with far less effort than before. 

## The Variational Auto Encoder (VAE)
The 'right'  latent space is the one that makes the distribution $p(\mathbf z| \mathbf \theta)$ the most likely to produce $\mathbf x$. We are therefore introducing a stage that complements the aforementioned _generative model or decoder_ given by $p(\mathbf x| \mathbf z ; \mathbf \theta) p(\mathbf z | \theta)$. 

This stage is called the _recognition model or encoder_ and is given by $p(\mathbf z| \mathbf x ; \mathbf \theta)$. The premise is this: the posterior $p(\mathbf z | \mathbf x ; \mathbf \theta)$ will result into a much more meaningful and compact latent space $\mathbf z$ than the prior $p(\mathbf z | \mathbf \theta)$. This encoding though, calls for sampling from a posterior that is itself intractable. We then need to use an approximation to such distribution: $q(\mathbf z| \mathbf x ; \mathbf \phi)$ and we call this the _inference model_ that approximates the recognition model and help us optimize the marginal likelihood. 

The VAE encoder-decoder spaces are clearly shown below. The picture shows the more compact space that is defined by the encoder. 

![vae](images/vae-spaces.png#center)
*VAE spaces and distributions (from [here](https://arxiv.org/pdf/1906.02691.pdf))*

The architecture of VAE includes four main components as shown below:

![vae](images/vae-architecture.png#center)
*VAE Architecture (from [here](https://arxiv.org/pdf/1906.02691.pdf))*

Similar to the generative model, the inference model can be, in general, a PGM of the form:

$$q(\mathbf z | \mathbf x ; \mathbf \phi) = \prod_{j=1}^M q(\mathbf z_j | Pa(\mathbf z_j), \mathbf x ; \mathbf \phi)$$

and this, similarly to the generative model, can be parametrized with a $DNN_{enc}(\phi)$. More specifically we obtain the approximation using the following construction:

$$ (\bm \mu,  \log \bm \Sigma ) = DNN_{enc}(\mathbf x, \bm \phi)$$
$$q(\mathbf z| \mathbf x ; \mathbf \phi) = N(\mathbf z; \bm \mu, \textsf{diag} \mathbf \Sigma) )$$

The $DNN_{enc}$ implements amortized variational inference, that is, it estimates the posterior parameters over a batch of datapoints and this offers significant boost in the parameter learning. 

Following the treatment in our [background probability chapter]({{<ref "../../ml-math/info-theory" >}}), we have met the concept of relative entropy or KL divergence that measures the "distance" between two distributions referenced on one of them. 

$$KL(q||p)= \mathbb{E}[\log q(\mathbf x) - \ln p(\mathbf x)] = - \sum_{\mathbf x} q(\mathbf x) \log \frac{p(\mathbf x)}{q(\mathbf x)}$$

We will use KL divergence to obtain a suitable loss function that will be used in the optimization of this approximation via the $DNN_{enc}$ network. Ultimately we are trying to minimize the KL divergence between the true posterior $p(\mathbf z| \mathbf x ; \mathbf \theta)$ and the approximate posterior $q(\mathbf z | \mathbf x ; \mathbf \phi)$  

$$KL(q(\mathbf z | \mathbf x ; \mathbf \phi) || p(\mathbf z | \mathbf \theta)) = - \sum_{\mathbf z}  q(\mathbf z | \mathbf x ; \mathbf \phi) \log \frac{p(\mathbf z | \mathbf x; \mathbf \theta))}{p(\mathbf z | \mathbf x ; \mathbf \phi)}$$
$$=  - \sum_{\mathbf z}  q(\mathbf z | \mathbf x ; \mathbf \phi) \log \frac{\frac{p(\mathbf z , \mathbf x; \mathbf \theta))}{p(\mathbf x)}}{q(\mathbf z | \mathbf x ; \mathbf \phi)} = - \sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) \log \Big[ \frac{p(\mathbf z , \mathbf x; \mathbf \theta))}{q(\mathbf z | \mathbf x ; \mathbf \phi)} \frac{1}{p(\mathbf x)}\Big]$$
$$=- \sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) \Big[ \log \frac{p(\mathbf z , \mathbf x; \mathbf \theta))}{q(\mathbf z | \mathbf x ; \mathbf \phi)} - \log p(\mathbf x) \Big] $$
$$=  -\sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) \log \frac{p(\mathbf z , \mathbf x; \mathbf \theta))}{q(\mathbf z | \mathbf x ; \mathbf \phi)} + \sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) \log p(\mathbf x) $$
$$=  -\sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) \log \frac{p(\mathbf z , \mathbf x; \mathbf \theta))}{q(\mathbf z | \mathbf x ; \mathbf \phi)} + \log p(\mathbf x) \sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) $$
$$= -\sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) \log \frac{p(\mathbf z , \mathbf x; \mathbf \theta))}{q(\mathbf z | \mathbf x ; \mathbf \phi)} + \log p(\mathbf x)$$
$$⇒\log p(\mathbf x) = KL(q(\mathbf z | \mathbf x ; \mathbf \phi) || p(\mathbf z | \mathbf \theta)) + \underbrace{\sum_{\mathbf z} q(\mathbf z | \mathbf x ; \mathbf \phi) \log \frac{p(\mathbf z , \mathbf x; \mathbf \theta))}{q(\mathbf z | \mathbf x ; \mathbf \phi)}}_{\text{L = Evidence Lower Bound (ELBO)}}$$

The bracketed $\mathcal L(q, \phi)$ quantity is called Evidence Lower Bound and is a functional of the distribution $q$ and a function of the parameters $\phi$. Why its a lower bound of the log-likelihood (evidence) function  $\log p(\mathbf x)$ and why its a useful quantity to consider?

Answering the last question first, we maximize the likelihood by effectively maximizing the $\mathcal L(q, \phi, \theta)$ since $KL(q(\mathbf z | \mathbf x ; \mathbf \phi) || p(\mathbf z | \mathbf \theta)) \ge 0$ by definition with zero only when $q(\mathbf z | \mathbf x ; \mathbf \phi) = p(\mathbf z | \mathbf \theta))$. Since

$$\mathcal L(q, \phi, \theta) =  \log p(\mathbf x) - KL(q(\mathbf z | \mathbf x ; \mathbf \phi) || p(\mathbf z | \mathbf \theta)) \le \log p(\mathbf x)$$. This is illustrated bellow:

![Bishop](images/Figure9.11.png#center)
*KL represents the tightness of the ELBO bound - From Bishop's book* 

As the figure above shows $KL(q(\mathbf z | \mathbf x ; \mathbf \phi) || p(\mathbf z | \mathbf \theta))$ represents the tightness of the ELBO $\mathcal L(q, \phi, \theta)$ since the closest the approximation becomes the smaller the gap between ELBO and the log likelihood. Maximizing the ELBO withe respect to $(\phi, \theta)$ will achieve "two birds with one stone" situation: it will maximize the marginal log likelihood that is used for data generation _and_ minimize the KL divergence improving the approximation in the encoder. On top of that, the ELBO allows joint optimization with respect to all the parameters $\phi$ and $\theta$ using SGD. This is described next via an example. 

### ELBO Optimization and MNIST Example
The following example is taken from the examples code of the excellent [Tensorflow Probability API](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py) and is very instructive. 

```python3
    """Trains a variational auto-encoder (VAE) on binarized MNIST.

    The VAE defines a generative model in which a latent code `Z` is sampled from a
    prior `p(Z)`, then used to generate an observation `X` by way of a decoder
    `p(X|Z)`. The full reconstruction follows

    ```none
    X ~ p(X)              # A random image from some dataset.
    Z ~ q(Z | X)          # A random encoding of the original image ("encoder").
    Xhat ~ p(Xhat | Z)       # A random reconstruction of the original image
                            #   ("decoder").
    ```

    To fit the VAE, we assume an approximate representation of the posterior in the
    form of an encoder `q(Z|X)`. We minimize the KL divergence between `q(Z|X)` and
    the true posterior `p(Z|X)`: this is equivalent to maximizing the evidence lower
    bound (ELBO),

    ```none
    -log p(x)
    = -log int dz p(x|z) p(z)
    = -log int dz q(z|x) p(x|z) p(z) / q(z|x)
    <= int dz q(z|x) (-log[ p(x|z) p(z) / q(z|x) ])   # Jensen's Inequality
    =: KL[q(Z|x) || p(x|Z)p(Z)]
    = -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
    ```

    -or-

    ```none
    -log p(x)
    = KL[q(Z|x) || p(x|Z)p(Z)] - KL[q(Z|x) || p(Z|x)]
    <= KL[q(Z|x) || p(x|Z)p(Z)                        # Positivity of KL
    = -E_{Z~q(Z|x)}[log p(x|Z)] + KL[q(Z|x) || p(Z)]
    ```

    The `-E_{Z~q(Z|x)}[log p(x|Z)]` term is an expected reconstruction loss and
    `KL[q(Z|x) || p(Z)]` is a kind of distributional regularizer. See
    [Kingma and Welling (2014)][1] for more details.

    This script supports both a (learned) mixture of Gaussians prior as well as a
    fixed standard normal prior. You can enable the fixed standard normal prior by
    setting `mixture_components` to 1. Note that fixing the parameters of the prior
    (as opposed to fitting them with the rest of the model) incurs no loss in
    generality when using only a single Gaussian. The reasoning for this is
    two-fold:

    * On the generative side, the parameters from the prior can simply be absorbed
        into the first linear layer of the generative net. If `z ~ N(mu, Sigma)` and
        the first layer of the generative net is given by `x = Wz + b`, this can be
        rewritten,

        s ~ N(0, I)
        x = Wz + b
            = W (As + mu) + b
            = (WA) s + (W mu + b)

        where Sigma has been decomposed into A A^T = Sigma. In other words, the log
        likelihood of the model (E_{Z~q(Z|x)}[log p(x|Z)]) is independent of whether
        or not we learn mu and Sigma.

    * On the inference side, we can adjust any posterior approximation
        q(z | x) ~ N(mu[q], Sigma[q]), with

        new_mu[p] := 0
        new_Sigma[p] := eye(d)
        new_mu[q] := inv(chol(Sigma[p])) @ (mu[p] - mu[q])
        new_Sigma[q] := inv(Sigma[q]) @ Sigma[p]

        A bit of algebra on the KL divergence term `KL[q(Z|x) || p(Z)]` reveals that
        it is also invariant to the prior parameters as long as Sigma[p] and
        Sigma[q] are invertible.

    This script also supports using the analytic KL (KL[q(Z|x) || p(Z)]) with the
    `analytic_kl` flag. Using the analytic KL is only supported when
    `mixture_components` is set to 1 since otherwise no analytic form is known.

    Here we also compute tighter bounds, the IWAE [Burda et. al. (2015)][2].

    These as well as image summaries can be seen in Tensorboard. For help using
    Tensorboard see
    https://www.tensorflow.org/guide/summaries_and_tensorboard
    which can be run with
    `python -m tensorboard.main --logdir=MODEL_DIR`

    #### References

    [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
        _International Conference on Learning Representations_, 2014.
        https://arxiv.org/abs/1312.6114
    [2]: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
        Autoencoders. In _International Conference on Learning Representations_,
        2015.
        https://arxiv.org/abs/1509.00519
    """

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import functools
    import os

    # Dependency imports
    from absl import flags
    import numpy as np
    from six.moves import urllib
    import tensorflow.compat.v1 as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions

    IMAGE_SHAPE = [28, 28, 1]

    flags.DEFINE_float(
        "learning_rate", default=0.001, help="Initial learning rate.")
    flags.DEFINE_integer(
        "max_steps", default=5001, help="Number of training steps to run.")
    flags.DEFINE_integer(
        "latent_size",
        default=16,
        help="Number of dimensions in the latent code (z).")
    flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
    flags.DEFINE_string(
        "activation",
        default="leaky_relu",
        help="Activation function for all hidden layers.")
    flags.DEFINE_integer(
        "batch_size",
        default=32,
        help="Batch size.")
    flags.DEFINE_integer(
        "n_samples", default=16, help="Number of samples to use in encoding.")
    flags.DEFINE_integer(
        "mixture_components",
        default=100,
        help="Number of mixture components to use in the prior. Each component is "
            "a diagonal normal distribution. The parameters of the components are "
            "intialized randomly, and then learned along with the rest of the "
            "parameters. If `analytic_kl` is True, `mixture_components` must be "
            "set to `1`.")
    flags.DEFINE_bool(
        "analytic_kl",
        default=False,
        help="Whether or not to use the analytic version of the KL. When set to "
            "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
            "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
            "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
            "then you must also specify `mixture_components=1`.")
    flags.DEFINE_string(
        "data_dir",
        default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
        help="Directory where data is stored (if using real data).")
    flags.DEFINE_string(
        "model_dir",
        default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
        help="Directory to put the model's fit.")
    flags.DEFINE_integer(
        "viz_steps", default=500, help="Frequency at which to save visualizations.")
    flags.DEFINE_bool(
        "fake_data",
        default=False,
        help="If true, uses fake data instead of MNIST.")
    flags.DEFINE_bool(
        "delete_existing",
        default=False,
        help="If true, deletes existing `model_dir` directory.")

    FLAGS = flags.FLAGS


    def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))


    def make_encoder(activation, latent_size, base_depth):
    """Creates the encoder function.

    Args:
        activation: Activation function in hidden layers.
        latent_size: The dimensionality of the encoding.
        base_depth: The lowest depth for a layer.

    Returns:
        encoder: A `callable` mapping a `Tensor` of images to a
        `tfd.Distribution` instance over encodings.
    """
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    encoder_net = tf.keras.Sequential([
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(4 * latent_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2 * latent_size, activation=None),
    ])

    def encoder(images):
        images = 2 * tf.cast(images, dtype=tf.float32) - 1
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                    _softplus_inverse(1.0)),
            name="code")

    return encoder


    def make_decoder(activation, latent_size, output_shape, base_depth):
    """Creates the decoder function.

    Args:
        activation: Activation function in hidden layers.
        latent_size: Dimensionality of the encoding.
        output_shape: The output image shape.
        base_depth: Smallest depth for a layer.

    Returns:
        decoder: A `callable` mapping a `Tensor` of encodings to a
        `tfd.Distribution` instance over images.
    """
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    decoder_net = tf.keras.Sequential([
        deconv(2 * base_depth, 7, padding="VALID"),
        deconv(2 * base_depth, 5),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5),
        conv(output_shape[-1], 5, activation=None),
    ])

    def decoder(codes):
        original_shape = tf.shape(input=codes)
        # Collapse the sample and batch dimension and convert to rank-4 tensor for
        # use with a convolutional decoder network.
        codes = tf.reshape(codes, (-1, 1, 1, latent_size))
        logits = decoder_net(codes)
        logits = tf.reshape(
            logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
        return tfd.Independent(tfd.Bernoulli(logits=logits),
                            reinterpreted_batch_ndims=len(output_shape),
                            name="image")

    return decoder


    def make_mixture_prior(latent_size, mixture_components):
    """Creates the mixture of Gaussians prior distribution.

    Args:
        latent_size: The dimensionality of the latent representation.
        mixture_components: Number of elements of the mixture.

    Returns:
        random_prior: A `tfd.Distribution` instance representing the distribution
        over encodings in the absence of any evidence.
    """
    if mixture_components == 1:
        # See the module docstring for why we don't learn the parameters here.
        return tfd.MultivariateNormalDiag(
            loc=tf.zeros([latent_size]),
            scale_identity_multiplier=1.0)

    loc = tf.compat.v1.get_variable(
        name="loc", shape=[mixture_components, latent_size])
    raw_scale_diag = tf.compat.v1.get_variable(
        name="raw_scale_diag", shape=[mixture_components, latent_size])
    mixture_logits = tf.compat.v1.get_variable(
        name="mixture_logits", shape=[mixture_components])

    return tfd.MixtureSameFamily(
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=tf.nn.softplus(raw_scale_diag)),
        mixture_distribution=tfd.Categorical(logits=mixture_logits),
        name="prior")


    def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(input=images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(input=images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


    def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.compat.v1.summary.image(
        name, pack_images(tensor, rows, cols), max_outputs=1)


    def model_fn(features, labels, mode, params, config):
    """Builds the model function for use in an estimator.

    Arguments:
        features: The input features for the estimator.
        labels: The labels, unused here.
        mode: Signifies whether it is train or test or predict.
        params: Some hyperparameters as a dictionary.
        config: The RunConfig, unused here.

    Returns:
        EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    del labels, config

    if params["analytic_kl"] and params["mixture_components"] != 1:
        raise NotImplementedError(
            "Using `analytic_kl` is only supported when `mixture_components = 1` "
            "since there's no closed form otherwise.")

    encoder = make_encoder(params["activation"],
                            params["latent_size"],
                            params["base_depth"])
    decoder = make_decoder(params["activation"],
                            params["latent_size"],
                            IMAGE_SHAPE,
                            params["base_depth"])
    latent_prior = make_mixture_prior(params["latent_size"],
                                        params["mixture_components"])

    image_tile_summary(
        "input", tf.cast(features, dtype=tf.float32), rows=1, cols=16)

    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    decoder_likelihood = decoder(approx_posterior_sample)
    image_tile_summary(
        "recon/sample",
        tf.cast(decoder_likelihood.sample()[:3, :16], dtype=tf.float32),
        rows=3,
        cols=16)
    image_tile_summary(
        "recon/mean",
        decoder_likelihood.mean()[:3, :16],
        rows=3,
        cols=16)

    # `distortion` is just the negative log likelihood.
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(input_tensor=distortion)
    tf.compat.v1.summary.scalar("distortion", avg_distortion)

    if params["analytic_kl"]:
        rate = tfd.kl_divergence(approx_posterior, latent_prior)
    else:
        rate = (approx_posterior.log_prob(approx_posterior_sample)
                - latent_prior.log_prob(approx_posterior_sample))
    avg_rate = tf.reduce_mean(input_tensor=rate)
    tf.compat.v1.summary.scalar("rate", avg_rate)

    elbo_local = -(rate + distortion)

    elbo = tf.reduce_mean(input_tensor=elbo_local)
    loss = -elbo
    tf.compat.v1.summary.scalar("elbo", elbo)

    importance_weighted_elbo = tf.reduce_mean(
        input_tensor=tf.reduce_logsumexp(input_tensor=elbo_local, axis=0) -
        tf.math.log(tf.cast(params["n_samples"], dtype=tf.float32)))
    tf.compat.v1.summary.scalar("elbo/importance_weighted",
                                importance_weighted_elbo)

    # Decode samples from the prior for visualization.
    random_image = decoder(latent_prior.sample(16))
    image_tile_summary(
        "random/sample",
        tf.cast(random_image.sample(), dtype=tf.float32),
        rows=4,
        cols=4)
    image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)

    # Perform variational inference by minimizing the -ELBO.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.cosine_decay(
        params["learning_rate"], global_step, params["max_steps"])
    tf.compat.v1.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo":
                tf.compat.v1.metrics.mean(elbo),
            "elbo/importance_weighted":
                tf.compat.v1.metrics.mean(importance_weighted_elbo),
            "rate":
                tf.compat.v1.metrics.mean(avg_rate),
            "distortion":
                tf.compat.v1.metrics.mean(avg_distortion),
        },
    )


    ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    FILE_TEMPLATE = "binarized_mnist_{split}.amat"


    def download(directory, filename):
    """Downloads a file."""
    filepath = os.path.join(directory, filename)
    if tf.io.gfile.exists(filepath):
        return filepath
    if not tf.io.gfile.exists(directory):
        tf.io.gfile.makedirs(directory)
    url = os.path.join(ROOT_PATH, filename)
    print("Downloading %s to %s" % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
    return filepath


    def static_mnist_dataset(directory, split_name):
    """Returns binary static MNIST tf.data.Dataset."""
    amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
    dataset = tf.data.TextLineDataset(amat_file)
    str_to_arr = lambda string: np.array([c == b"1" for c in string.split()])

    def _parser(s):
        booltensor = tf.compat.v1.py_func(str_to_arr, [s], tf.bool)
        reshaped = tf.reshape(booltensor, [28, 28, 1])
        return tf.cast(reshaped, dtype=tf.float32), tf.constant(0, tf.int32)

    return dataset.map(_parser)


    def build_fake_input_fns(batch_size):
    """Builds fake MNIST-style data for unit testing."""
    random_sample = np.random.rand(batch_size, *IMAGE_SHAPE).astype("float32")

    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            random_sample).map(lambda row: (row, 0)).batch(batch_size).repeat()
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            random_sample).map(lambda row: (row, 0)).batch(batch_size)
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    return train_input_fn, eval_input_fn


    def build_input_fns(data_dir, batch_size):
    """Builds an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    def train_input_fn():
        dataset = static_mnist_dataset(data_dir, "train")
        dataset = dataset.shuffle(50000).repeat().batch(batch_size)
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    # Build an iterator over the heldout set.
    def eval_input_fn():
        eval_dataset = static_mnist_dataset(data_dir, "valid")
        eval_dataset = eval_dataset.batch(batch_size)
        return tf.compat.v1.data.make_one_shot_iterator(eval_dataset).get_next()

    return train_input_fn, eval_input_fn


    def main(argv):
    del argv  # unused

    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    if FLAGS.delete_existing and tf.io.gfile.exists(FLAGS.model_dir):
        tf.compat.v1.logging.warn("Deleting old log directory at {}".format(
            FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)

    if FLAGS.fake_data:
        train_input_fn, eval_input_fn = build_fake_input_fns(FLAGS.batch_size)
    else:
        train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir,
                                                        FLAGS.batch_size)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_steps=FLAGS.viz_steps,
        ),
    )

    for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)


    if __name__ == "__main__":
    tf.compat.v1.app.run()
```