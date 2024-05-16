---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

## I. Introduction

Generative AI is the closest we have to pure magic. Large Language Models (LLM) can now seamlessly chat with us, while other models create songs, images, and more recently even videos from a single text prompt. Motivated by my selfish desire to demystify the magic behind, this survey focuses on a group of models mostly utilized for the latter task, called Diffusion Models.

My main motivation here is somewhat selfish. As I set out to learn, understand, and demystify the magic, I found that while there are some excellent resources (beyond the original papers) blog posts (see e.g. ), videos, etc, there were some details never properly explained and implementations I found were often hard to access. In addition, I know it from my academic past that you rarely learn something as well as when you teach it for the first time.

## II. Generative AI

In abstract form, the task of Generative AI can be phrased deceptively simply:

> Given a highly complex distribution $q(x)$ that lives in a high dimensional ambient space but expected to be concentrated on a much lower dimensional, structured, yet intractable manifold, can we learn the structure of this distribution from a possibly large but nevertheless finite set of examples and produce a reasonable approximation $p(x)$.

For some typical example domains, a second of standard sample rate (44.1kHz) audio can be described by an amplitude vector $x\in\mathbb{R}^{44100}$. A 12 Megapixel RGB image taken by a modern iPhone is representing by an intensity vector in $\mathbb{R}^{3\times 4000\times 3000}$ which uses 36M dimensions. A second of 4K RGB video at 60Hz refresh rate lives in $\mathbb{R}^{3\times 3840\times 2160\times 60}$ which is a ~1.5B dimensional space.

It is not hard to imagine that the overwhelming majority of points in these spaces correspond to absolute jibberish and the meaningful media content lies on a vastly lower dimensional submanifold, the famous manifold hypothesis. We can hardly hope to be able to characterize this submanifold exactly, let alone tractably describe probability distributions over them, however, we can reasonably hope that ML models can learn them with lot less impediment from the curse of dimensionality than the ambient dimension would suggest.

Once we learn the distribution $p(x)$ can then use this learned distribution to (1) sample new (perhaps unseen) examples; (2) detect when a new example is unlikely to have come from the learned distribution or (3) use it in some other downstream task. 

As always, the choice of metric to measure the quality of the fit can, in general depend on the particular application at hand. In these notes, we will mostly use KL-divergence (or relative entropy)

$$ D(q \| p) = \int q(x)\log\frac{q(x)}{p(x)} dx $$

to capture the dissimilarity of $q(x)$ and $p(x)$. As it is well known, the KL-divergence is not symmetric, that is $D(q \| p) \neq D(p \| q)$. The reason behind the particular choice of $D(q \| p)$ is that the integrand above is non-zero over the empirical data distribution observed in the training set. Note however, that it is not possible to evaluate the KL-divergence as is, due to the very fact that $q(x)$ is unknown. Thus instead, we note that

$$ D(q \| p) = CE(q \| p) - H_q(X),$$

where the entropy $H_q(X) = -\int q(x)\log q(x) dx$ does not depend on $p(x)$ and does not need to be optimized. On the other hand, the cross-entropy 

$$CE(q \| p) = -\int q(x) \log p(x) dx$$

can be approximated by Monte-Carlo sampling as the average negative log-likelihood of the data under the model:

$$ CE(q \| p) \approx -\frac{1}{|data|}\sum_{i\in data}\log p(x_i)$$

This, of course, comes at the price that we never know how far from the optimum we are with a particular model.

Now we only need to learn $p(x_i)$ and sample from it, there are several mainstream approaches to do this: 

- **Autoregressive models**, e.g. most LLM-s, that factorize $p(x)=\prod_ip(x_i\|x_{<i})$ into the product of conditional probabilities where the next "token" is determined by looking at the previous "tokens" according to some (sometimes arbitrary, sometimes natural) ordering.

- **Generative Adversarial Networks (GAN)**, where a generator model is trained to fool a discriminator model which is trained to tell generated and real images apart.

- **Variational Autoencoders (VAE)**, which is a latent variable model that jointly learns to approximate the  optimal latent distribution using an encoder and to decode the latent into an image.

- **Flow models**, [...]

- **Diffusion models**, the subject of this piece, that are similar to VAEs, however, their latent variable is the path of a diffusion process and the prior latent distribution is both learnable and tractable.

## III. Unconditioned generation with Diffusions

As with any other type of generative models, [...]

### (a) The forward process

As described above, Diffusion Models learn how to denoise by studying how to noise. The proposal of the seminal paper [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) is to progressively add more and more isotropic Gaussian noise to a training image until the information is nearly destroyed. In particular, the noising is done through a Markov-Chain with a Gaussian transition kernel

$$ q(x_t|x_{t-1}) = \mathcal{N}(x_{t-1}, \sqrt{1-\beta_t}, \beta_t I) $$

or what one does in practice, use the Stochastic *Difference* Equation (also known as the *reparametrization trick*)

$$ {\bf x}_t = \sqrt{1-\beta_t}{\bf x}_{t-1} + \sqrt{\beta_t}\varepsilon_t, \qquad \varepsilon_t\sim\mathcal{N}(0, I)\qquad (*)$$

[Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) keeps it more general to allow for a binomial diffusion model, but the ideas are the same. For simplicity, we focus on the Gaussian case.

#### **Variance balance**

In most blog posts, and even in the original paper, the $\sqrt{1-\beta_t}$ term just falls out of the sky, however, this choice is critical. To see why, first note that this process proceeds pixelwise and if we set $\sigma_t$ to be the variance of a single pixel, we get

$$ \sigma_t^2 = (1-\beta_t)\sigma_{t-1}^2 + \beta_t $$

As we will see, we want $x_t$ to converge to a (pixel-wise) stationary Gaussian distribution after a possibly large number of iterates. In this case, the variance should stabilize to some $\sigma* and thus taking a large enough $t$,

$$ \sigma^2 \approx (1-\beta_t)\sigma^2 + \beta_t, \qquad\textrm{or} \qquad \sigma^2\approx 1.$$

Note that $\sigma^2=1$ is actually an exact stationary variance, in fact, the only one. Basically, if the coefficient of ${\bf x_{t-1}}$ was larger, the variances injected at each step could pile up, while if it was smaller, the variability could collapse (although keep in mind that $\sigma_t\geq \beta_t$).

#### **Convergence to Gaussian stationary distribution**

To take idea this further, let us borrow the notation from [Ho et al (2020)](https://arxiv.org/pdf/2006.11239):

$$ \alpha_t = 1-\beta_t,\qquad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s $$

which allows us to unfold (*) to

$$ {\bf x}_t = \sqrt{\bar{\alpha}_t}{\bf x}_0 + \sqrt{1-\bar{\alpha}_t}\bar{\varepsilon}_t, $$

where $\bar{\varepsilon}_t$ are still isotropic standard Gaussians, but they are not independent anymore for different $t$. The reader is invited to prove this by induction, clearly, the choice of $1-\beta_t$ is crucial for this to hold.

So what is the distribution of $\bf{x}_t$? Clearly, for fixed $\bf{x}_0$ it is a Gaussian, but recall that our starting point is that $\bf{x}_0$ comes from some distribution $q(x_0)$ we are trying to learn and that is far from Gaussian. Taking that into account (which corresponds to sampling an example from $q$ followed by running the Markov chain), $\bf{x}_t$ is an intractable mixture of Gaussians with mixing distribution $q(x_0)$. For large $t$, however, as long as $\bar{\alpha}_t\to 0$ (which translates to $\beta_t$ is not being too small), we have $\bf{x_t} \approx \bar{\varepsilon}_t \sim \mathcal{N}(0, I)$.  This fact can be made rigorous e.g. by using [characteristic functions](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)), and is the special case of a more general result on the ergodicity of Markov chains. Furthermore, if $\beta_t > c > 0$ for some number $c$, then $\bar{\alpha}_t\leq (1-c)^t$ and the convergence is exponentially fast.

To summarize, the Markov chain morphs the initial intractable distribution $q(x_0)$ to an approximately Gaussian $q(x_T)$ which we can very easily handle analytically and sample from. This convergence of distributions happens faster with larger $\beta$-s. The question is now whether we can learn from this process about how to go the other way and transform a standard isotropic Gaussian distribution to an approximation $p(x_0)$ of $q(x_0)$.

### (b) The backward process and the model probability

It will be helfpul to think in analogy with VAE-s where the diffusion trajectory $z = x_{1:T}$ is for all intents and purposes a complex hidden variable. Then $q(x_{1:T} | x_0)$ plays the role of the encoder and we are looking to learn a decoder $p_{\theta}(x_0|x_{1:T})$. In what follows, we will use the VAE analogy often. Note, however, that it is not a perfect one as we do not have an explicit, fixed prior for the full latent $z$ only for $p(x_T)$. On the other hand, this is exactly what allows for a complex, but still tractable full prior $p_\theta(x_{1:T})$.

We are going to look for this full prior and the decoded probability in the form of a Markov process working backwards on the diffusion trajectory. In particular, we start with a Gaussian $p(x_T) \sim \mathcal{N}(x_T; 0, I)$ at time $T$ and proceed backward according to a parametrized transition kernel $p_{\theta}(x_{t-1} | x_t)$ that we will learn. Eventually, we arrive at the model probability

$$ p_\theta(x_0) = \int p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)dx_{1:T}. $$

In the VAE language, this corresponds to $p(x_0) = \int p(x_0, z)dz$, computing the model probability by enumerating all possible latents that could produced $x_0$.

As a result, this integral is usually highly intractable but the VAE analogy guides us towards utilizing the forward trajectory in an importance sampling scheme to emphasize plausible latent values. In other words, the latent proposal distribution is chosen to be $q(x_{1:T}|x_0) = \prod_{i=1}^Tq(x_t|x_{t-1})$, that is, the law of the forward trajectory launched from the starting point $x_0. With this,

$$
p(x_0) = \int q(x_{1:T}|x_0)p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} dx_{1:T} = \mathbb{E}_{x_{1:T}\sim q(\cdot|x_0)}p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})},
$$

which we can evaluate by simple Monte-Carlo by sampling by simulating the forward process. We are, however, not done yet, we haven't even use the fact that the model probability $p(x_0)$ should resemble the data distribution $q(x_0)$.

### (c) The learning objective

Again, just as in VAE-s, we would like $p_{\theta}(x_0)$ to approximate $q(x_0)$ in KL-divergence which we have seen to be equivalent to minimizing their cross-entropy $CE(q, p_{\theta})$:

$$
\begin{align*}
-\int q(x_0)\log p_\theta(x_0)dx_0 = - \mathbb{E}_q\log p_{\theta}(x_0) = -\mathbb{E}_q\log \mathbb{E}_{x_{1:T}\sim q(\cdot|x_0)}\left[p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]
\end{align*}
$$

where the inner expectation is over the diffusion trajectories with fixed $x_0$ while the outer expectation is over $x_0$. If we wanted to naively evaluate this, we could sample $x_0$ from our datase, simulate $x_{1:T}$ from $q$ for each sample, and use Monte-Carlo to turn the expectation over averages. The problem is that this would be a potentially heavily biased estimator due to the presence of the logarithm (try computing the mean of the estimator, you will get stuck at $\mathbb{E}\log\sum_{samples}$).  

Instead, the VAE recipe suggests we use Jensen's inequality to swap the logarithm and the inner expectation and obtain the usual *evidence lower bound (ELBO)* ("lower" refers to the viewpoint of maximizing the likelihood where there are no minus signs and the inequality is flipped):

$$
-\int q(x_0)\log p_\theta(x_0)dx_0 \leq -\mathbb{E}_q\log \left[p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right] =: L
$$

where the expectation is now taken over the full trajectory $x_{0:T}$. 
Since Jensens was only applied to the inner expectation, equality here holds if and only if the term under the log is the same for all trajectories $x_{1:T}$ which implies $q(x_{1:T}|x_0)\sim p_\theta(x_{1:T}, x_0)$. After normalization, this translates to $q(x_{1:T}|x_0) = p_\theta(x_{1:T}| x_0)$ that is when the distribution of the forward trajectory is exactly the posterior of $x_{1:T}$ under $p_{\theta}$ given $x_0$.

This exact posterior is, of course, no easier to compute than the model probability itself, and VAEs try to jointly optimize the encoder $q(z|x)$ and the decoder $p(x|z)$ to simultaneously close the Jensen-gap and the optimize the bound. In our situation, however, the distribution of the forward process is fairly rigidly prescribed with the $\beta_t$-s being the only (potentially) learnable *variational parameters*. The hope is that the trainable full prior will bring us closer to this fixed $q$ instead.

At this point one could provide the parametrization for $p_{\theta}$, evaluate with Monte-Carlo, take gradients and train as usual. This sampling, however is fairly expensive and it turns out we can do better. Let us first separate the edge terms $t = 0$ and $t=T$ in $L$: 

$$
L = -\mathrm{E}_q\log p(x_T) -\sum_{t=2}^T\mathrm{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} - \mathbb{E}_q\log\frac{p_\theta(x_0 | x_1)}{q(x_1|x_0)}
$$


## Conditioned generation



## Architectures%   

## Resources

Besides the references in the text, I took inspiration from [Lilian Wang's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process) and the Stanford [Deep Generative Models course](https://deepgenerativemodels.github.io/).