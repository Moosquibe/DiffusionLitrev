---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

## I. Introduction

In my opinion, Generative AI is the closest we have to pure magic. Large Language Models (LLM) can now seamlessly chat with us, while other models create songs, images, and more recently even videos from a single text prompt. Motivated by my selfish desire to demystify the magic behind, this survey focuses on a group of models mostly utilized for the latter task, called Diffusion Models.

My main motivation here is somewhat selfish. As I set out to learn, understand, and demystify the magic, I found that while there are some excellent resources (beyond the original papers) blog posts (see e.g. ), videos, etc, there were some details never properly explained and implementations I found were often hard to access. In addition, I know it from my academic past that you rarely learn something as well as when you teach it for the first time.

## II. Generative AI

In abstract form, the task of Generative AI can be phrased deceptively simply:

> Given a highly complex distribution $q(x)$ that lives in a high dimensional ambient space but expected to be concentrated on a much lower dimensional, structured, yet intractable manifold, can we learn the structure of this distribution from a possibly large but nevertheless finite set of examples and produce a reasonable approximation $p(x)$.

For example, a standard sample rate (44.1kHz) audio, a second of music can be described by an amplitude vector $x\in\mathbb{R}^{44100}$. A 12 Megapixel RGB image taken by a modern iPhone is representing by an intensity vector in $\mathbb{R}^{3\times 4000\times 3000}$ which uses 36M dimensions. A second of 4K RGB video at 60Hz refresh rate lives in $\mathbb{R}^{3\times 3840\times 2160\times 60}$ which is a ~1.5B dimensional space. 

It is not hard to imagine that the overwhelming majority of points in these spaces correspond to absolute jibberish and the meaningful media content lies on a vastly lower dimensional submanifold, the famous manifold hypothesis. We can hardly hope to be able to characterize this submanifold exactly, let alone tractably describe probability distributions over them, however, we can reasonably hope that ML models can learn them with lot less impediment from the curse of dimensionality than the ambient dimension would suggest.

Once we learn the distribution $p(x)$ can then use this learned distribution to (1) sample new (perhaps unseen) examples; (2) detect when a new example is unlikely to have come from the learned distribution or (3) use it in some other downstream task. 

As always, the choice of metric to measure the quality of the fit can, in general depend on the particular application at hand. In these notes, we will mostly use KL-divergence (or relative entropy)

$$ D(q \| p) = -\int q(x)\log\frac{q(x)}{p(x)} dx $$

to capture the dissimilarity of $q(x)$ and $p(x)$. As it is well known, the KL-divergence is not symmetric, that is $D(q \| p) \neq D(p \| q)$. The reason behind the particular choice of $D(q \| p)$ is that the integrand above is non-zero over the empirical data distribution observed in the training set. Note however, that it is not possible to evaluate the KL-divergence as is, due to the very fact that $q(x)$ is unknown. Thus instead, we note that

$$ D(q \| p) = - H_q(X) + CE(q \| p),$$

where the entropy $H_q(X) = -\int q(x)\log q(x) dx$ does not depend on $p(x)$ and does not need to be optimized. On the other hand, the cross-entropy $CE(q \| p) = \int q(x) \log p(x) dx$ can be approximated by Monte-Carlo sampling as the average log-likelihood of the data under the model:

$$ CE(q \| p) \approx \frac{1}{|data|}\sum_{i\in data}\log p(x_i)$$

This, of course, comes at the price that we never know how far from the optimum we are with a particular model.

Now we only need to learn $p(x_i)$ usually through some parametric model $p_{\theta}(x)$. In very rough terms, Diffusion Models learn how to create "art" from noise by learning from observing how art can be morphed into noise. In this regard, they are different from autoregressive models, like most LLM-s, that factorizes $p(x)=\prod_ip(x_i\|x_{<i})$ into the product of conditional probabilities where the "next token" is determined by looking at the "previous tokens" according to some (sometimes arbitrary, sometimes natural) ordering. They are also different from Generative Adversarial Netwoks (GANs), where a generator model is trained to fool a discriminator model which is trained to tell generated and real images apart. [VAE, Flow based models]

## III. Unconditioned generation with Diffusions

As with any other type of generative models, [...]

### (a) The forward process

As described above, Diffusion Models learn how to denoise by studying how to noise. The proposal of the seminal paper [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) is to progressively add more and more isotropic Gaussian noise to a training image until the information is nearly destroyed. In particular, the noising is done through a Markov-Chain with a Gaussian transition kernel

$$ q(x_t|x_{t-1}) = \mathcal{N}(x_{t-1}, \sqrt{1-\beta_t}, \beta_t I) $$

or what one does in practice, use the Stochastic *Difference* Equation (also known as the *reparametrization trick*)

$$ {\bf x}_t = \sqrt{1-\beta_t}{\bf x}_{t-1} + \sqrt{\beta_t}\varepsilon_t, \qquad \varepsilon_t\sim\mathcal{N}(0, I)\qquad (*)$$

The paper keeps it more general to allow for a binomial diffusion model. For simplicity, we focus on the Gaussian case.

#### **Variance balance**

In most blog posts, and even in the original paper, the $\sqrt{1-\beta_t}$ term just falls out of the sky, however, this choice is critical. To see why, first note that this process proceeds pixelwise and if we set $\sigma_t$ to be the variance of a single pixel, we get

$$ \sigma_t^2 = (1-\beta_t)\sigma_{t-1}^2 + \beta_t $$

As we will see, we want $x_t$ to converge to a (pixel-wise) stationary Gaussian distribution after a possibly large number of iterates. In this case, the limit $\sigma = \lim_{t\to\infty}\sigma_t$ exists and taking a large enough $t$,

$$ \sigma^2 \approx (1-\beta_t)\sigma^2 + \beta_t, \qquad\textrm{or} \qquad \sigma^2\approx 1.$$

Note that $\sigma^2=1$ is actually an exact stationary variance, in fact, the only one. Basically, if it was larger, the variances injected at each step could pile up, while if it was smaller, the variability could collapse (although keep in mind that $\sigma_t\geq \beta_t$).

#### **Convergence to Gaussian stationary distribution**

To take this further, let us borrow the notation from [Ho et al (2020)](https://arxiv.org/pdf/2006.11239):

$$ \alpha_t = 1-\beta_t,\qquad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s $$

which allows us to unfold (*) to

$$ {\bf x}_t = \sqrt{\bar{\alpha}_t}{\bf x}_0 + \sqrt{1-\bar{\alpha}_t}\bar{\varepsilon}_t, $$

where $\bar{\varepsilon}_t$ are still isotropic standard Gaussians, but they are not independent anymore for different $t$. The reader is invited to prove this by induction, clearly, the choice of $1-\beta_t$ is crucial for this to hold.

So what is the distribution of $\bf{x}_t$? Clearly, for fixed $\bf{x}_0$ it is a Gaussian, but recall that our starting point is that $\bf{x}_0$ comes from some distribution $q(x_0)$ we are trying to learn and that is far from Gaussian. Taking that into account (which corresponds to sampling an example from $q$ followed by running the Markov chain), $\bf{x}_t$ is an intractable mixture of Gaussians with mixing distribution $q(x_0)$. For large $t$, however, as long as $\bar{\alpha}_t\to 0$ (which translates to $\beta_t$ is not being too small), we have $\bf{x_t} \approx \bar{\varepsilon}_t \sim \mathcal{N}(0, I)$.  This fact can be made rigorous e.g. by using [characteristic functions](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)), and is the special case of a more general result on the ergodicity of Markov chains. Furthermore, if $\beta_t > c > 0$ for some number $c$, then $\bar{\alpha}_t\leq (1-c)^t$ and the convergence is exponentially fast.

To summarize, the Markov chain morphs the initial intractable distribution $q(x_0)$ to an approximately Gaussian one $q(x_T)$ which we can very easily handle analytically and sample from. This convergence of distributions happens faster with larger $\beta$-s. The question is now whether we can learn from this process about how to go the other way and transform a standard isotropic Gaussian distribution to an approximation $p(x_0)$ of $q(x_0)$. Success on this task would mean that we have a way to approximately evaluate $q(x_0)$ and sample from it by simply sampling from a Gaussian.

### (b) The backward process and the objective

Following up on the previous section, we try to learn a Markov chain that starts at a Gaussian $p(x_T) \sim \mathcal{N}(x_T; 0, I)$ and proceeds according to the parametrized transition kernel $p_{\theta}(x_{t-1}\|x_t)$ to eventually arrive at

$$ p_\theta(x_0) = \int p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)dx_{1:T} $$

approximating $q(x_0)$ in the sense of the cross-entropy $ CE = \int q(x_0)\log p_\theta(x_0)dx_0 $

### (c) The step-size schedule



## Conditioned generation

## Architectures%   