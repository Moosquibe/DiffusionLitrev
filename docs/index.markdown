---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

:construction:

## I. Introduction

Generative AI is about how we can conjure different media from our imagination and a little noise and is thus the closest thing we have to magic. Large Language Models (LLM) can now seamlessly chat with us, while other models create songs, images, or even videos from a single text prompt. Motivated by the desire to peek behind the curtain and demystify the magic behind modern generative AI, this survey focuses on a group of models giving the state of the art in image and video sythesis called Diffusion Models.

As mentioned, my main motivation here is to teach myself how the sausage is made, you rarely learn something as well as when you teach it for the first time. While there is no shortage of often excellent self-teach resources, including the original papers, blog posts YouTube videos, etc, I found that some details are not properly explained anywhere, implementations are often hard to access and accessible holistic overviews are rare as well. These notes therefore attempt to fill the void by providing a deep dive, zero-to-hero style overview.

When I have the time, I plan to provide educationally motivated PyTorch implementations in the associated [GitHub Repository](https://github.com/Moosquibe/DiffusionLitrev). This will also allow me to do my own experiments and comparisons on which I will report in a section below.

## II. Generative AI

In abstract form, the task of Generative AI can be phrased deceptively simply:

> Given a highly complex, high dimensional distribution $q(x)$ can we learn the structure of this distribution from a possibly large but nevertheless finite set of examples and produce a reasonable approximation $p_\theta(x)\approx q(x)$.

For some typical example domains, a second of standard sample rate (44.1kHz) audio can be described by an amplitude vector $x\in\mathbb{R}^{44100}$. A 12 Megapixel RGB image taken by a modern iPhone is representing by an intensity vector in $\mathbb{R}^{3\times 4000\times 3000}$ which uses 36M dimensions. A second of 4K RGB video at 60Hz refresh rate lives in $\mathbb{R}^{3\times 3840\times 2160\times 60}$ which is a ~1.5B dimensional space.

It is not hard to imagine that the overwhelming majority of points in these spaces correspond to jibberish and the meaningful media content lies on a vastly lower dimensional submanifold. This is the famous *manifold hypothesis*. We can hardly hope to be able to characterize this submanifold exactly, let alone tractably describe probability distributions over them, however, we can reasonably hope that ML models can learn them with lot less impediment from the curse of dimensionality than the ambient dimension would suggest.

Once we learn the distribution $p_\theta(x)$ can then use this learned distribution to

1. **Sample** new instances from $p_\theta(x)$;
2. **Estimate the density/probability** at a datapoint $x$ and detect whether it is unlikely to have come from the learned distribution;
3. Use the learnt distribution to make other **probabilistic queries** of interest;
4. Often in the process of learning $p(x)$, we gain some insight into the structure in the form of latent variables. This can then be used of **unsupervised clustering/representation learning**, etc.

### **Types of generative models**

[TODO: Citations]

There is a rich variety of models that can be utilized for the generative learning task. There are three main families:

1. **Likelihood based models** which directly learn the probability mass/density function by maximizing likelihood. The main challenge is to keep the normalizing constant (keeping $p(x)$ a probability) tractable, which necessitates either strong restrictions on the model architectures or must rely on surrogate objectives to approximate maximum likelihood training. The most common subtypes are:

    - **Autoregressive models**, e.g. most LLM-s, that factorizes the probability into the product of conditional probabilities according to some sometimes natural, sometimes arbitrary ordering: $p(x)=\prod_ip(x_i\|x_{<i})$. The model then learns these conditional distributions (and some seeding distribution for $x_0$). This structure makes the evaluation of the likelihood easy, but the generation has to be sequential which can be very slow.
    - **Variational Autoencoders (VAE)**, are latent variable models with a training methodology that jointly learns (1) what latent values are plausible for a given observation (encoder) and (2) how to decode a given value of the latent into a distribution over possible data domain instances. It does this to simultaneously optimize a surrogate loss function and close the gap between this surrogate and the likelihood.
    - **Normalizing flow models** that learn invertible deterministic mappings between the latent space and the observation domain and uses it to transform the latent densities to densities in the data domain. This allows computation of the likelihood through the simple prior distribution for the latent but needless to say, the invertibility puts a fair amount of restriction of the architecture of this mapping is restricting to continuous distributions.
    - **Energy based models** model the logarithm of the unnormalized likelihood $E_\theta(x)$ in $p_{\theta}(x) = Z_\theta^{-1}\exp E_\theta(x)$ and work around the limitations of the unknown normalization constant by contrastive learning techniques that basically all boil down to pushing $E_{\theta}(x)$ up on data samples and push down on contrasting negative samples without collapsing the energy landscape.
    - **Diffusion models** that are latent variable models, however, the latent is the path of a diffusion process and the prior latent distribution is both learnable. They can also be viewed as stochastic flows or stacked VAE-s and use a similar surrogate objective approach. They are the subject of this survey and we will go into the topic in much more detail below.

2. **Implicit generative models** where the probability distribution is implicitly represented by its sampling process but without a tracktable density/mass function.

   - **Generative Adversarial Networks (GAN)**, the state of the art for quite some time, where a generator model is trained to fool a discriminator model which is trained to tell generated and real images apart. They have excellent generation properties and were the state of the art in sample quality for quite a while generating lots of research interest (see the [GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)).
   
        On the other hand they are notoriously hard to train due to the nature of the min-max adversarial training. Indeed, they look for a good saddle point instead of an optimum and the objective function thus oscillates heavily. Often either the generator or the discriminator simply "wins" (model collapse), or the model start oscillating between modes of the data distribution (mode collapse) and the training fails. There is a large [bag of tricks](https://github.com/soumith/ganhacks) employed in practice. These difficulties eventually caused them to mostly fall out of the mainstream in favor of diffusion models.

3. **Score based generative models** that circumvents the difficulties of estimating the normalization constant by instead modeling the gradient of the log-probability often referred to as the Stein score. This is a vector field pointing towards increasing log-probability and can be learned by a model through a procedure called score matching. This vector field can also be used to guide a randomized process called Langevin dynamics to extract samples.

They have achieved state of the art performance at the time on many tasks and later it was discovered that they are very tightly related to diffusion models in that both can be viewed as discretization of Stochastic Differential Equations (SDE) driven by score functions. In short, diffusion models and score based models turned out to be two viewpoints of the same thing.

### Evaluating generative models

There are several ways to measure how "far" two probability distributions are, and the particular choice could depend on the application at hand. Indeed, [Theis et al, 2016](https://arxiv.org/pdf/1511.01844) showed that unless the true data generating process is a member of the model class, which virtually never happens in practice, different objective functions can lead to very different results. Understanding the trade-offs is thus crucial for choosing the right metric for the application at hand. Moreover, it is important not to take good performance in one application as evidence of similar performance in another one. For example, [Theis et al, 2016](https://arxiv.org/pdf/1511.01844) shows that high likelihood and realistic image synthesis properties do not imply each other at all.

It is usually uncommon to discuss metrics first, but I find that during the exposition dump of a large sequence of different modelling approaches, it is hard to keep track of how to compare them. This is of course a non-exhaustive list as new metrics are invented every day.

#### **Kullback-Leibler divergence**

The most natural criterion to judge performance is how well the model distribution $p(x)$ matches the distribution of $q(x)$ of the data.  However, the most common choice is to use KL-divergence (also known as relative entropy) due to its connection to the likelihood function (thus especially favored by likelihood based learning). It is defined as

$$ D(q \| p) = \int q(x)\log\frac{q(x)}{p(x)} dx $$

and measures the dissimilarity of $q(x)$ and $p(x)$. It is not symmetric, $D(q \| p) \neq D(p \| q)$,the reason behind the particular choice here is that the integrand above will then be encouraged to be non-zero over the empirical data distribution observed in the training set. The first challenge about KL-divergence is that we cannot evaluate it due to the very fact that $q(x)$ (note that even sampling wouldn't help due to $\log q(x)$). However, note that

$$ D(q \| p) = CE(q \| p) - H_q(X),$$

where the entropy $H_q(X) = -\int q(x)\log q(x) dx$ does not depend on $p(x)$ and does not need to be optimized. On the other hand, the cross-entropy

$$CE(q \| p) = -\int q(x) \log p(x) dx$$

can now be approximated by Monte-Carlo sampling as the average negative log-likelihood of the data under the model distribution:

$$ CE(q \| p) \approx -\frac{1}{N}\sum_{i=1}^N\log p(x_i)$$

where $N$ is the size of the dataset. This, of course, comes at the price that we never know how far from the optimum we are with a particular model.

Drawbacks of likelihood as a metric include that high likelihood models can have arbitrarily terrible generated samples as pointed out in [Theis et al, 2016](https://arxiv.org/pdf/1511.01844) and vica versa.

#### **Fisher divergence**

In some models, for example ones based on score matching, we cannot tracktably compute the normalization constant and therefore the full probability. In this case, we can resort to computing their scores

$$ J(\theta) = \frac{1}{2}\mathbb{E}_q\|s_p(x; \theta) - s_q(x)\|^2 $$

where $s_p(x;\theta) = \nabla_x\log p_\theta(x)$ is the score of the model distribution and similarly, $s_q(x) = \nabla_x\log q(x)$ is the data distribution score. Computing the latter is clearly challenging, fortunately [Hyvarinen, 2005](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) showed that a clever integration trick can turn this into the computable

$$ J(\theta) = \mathbb{E}_q\left[\mathrm{Tr}(\nabla_x s_p(x; \theta))+\frac{1}{2}\|s_p(x;\theta)\|^2\right] $$

For deep neural network models, the computation of the trace is fairly prohibited since all $D$ diagonal entries have to be computed through backpropagation. [Song et al 2019a](https://arxiv.org/pdf/1905.07088) proposed to mitigate this by replacing the full score comparison with comparing random, one dimensional projections which leads to the *sliced score matching* matching objective:

$$ J(\theta) = \mathbb{E}_{v\sim p_v}\mathbb{E}_q\left[v^T\nabla_x s_p(x; \theta)v+\frac{1}{2}(v^T s_p(x;\theta))^2\right], $$

where $p_v$ is a Rademacher (uniform on the vertices of the $D$ dimensional hypercube $\\{\pm 1\\}^D$) or an isotropic stadard normal.

#### **Inception Score (IS)**

IS, introduced by [Salimans et al., 2016](https://arxiv.org/pdf/1606.03498) to evaluate GAN-s, is a metric that evaluates generation by how well the generated images match the real images in the dataset in terms of their diversity and quality using statistic of the popular Inception v3 deep convolutional network designed for classification tasks on ImageNet. For a given image, it produces a vector of conditional probabilities $p_i(y\|x)$ over the thousand Image labels.

IS is then defined as the exponential of the expected KL divergence between the conditional and the marginal distribution under the generative distribution:

$$ IS = \exp(\mathbb{E}_{x\sim p_{\theta}}D_{KL}(p_{i}(y|x)\|p_i(y))),$$

where $p_i(y) = \mathbb{E}_{x\sim p_\theta}p_i(y\|x)$ is the marginal class distribution. To see what this metric is trying to accomplish, note that some easy manipulation shows

$$ \log IS = I(y;x) = H(y)-H(y|x),$$

that is, the logarithm of the IS is the mutual information between the class distribution and the generated sample. By the decomposition in the final step, a high IS corresponds to high entropy for the class distribution encouraging diversity, and low entropy for the Inception model's output encouraging clear images with a single object.

Besides the obvious limitation to labeled dataset, [Barratt & Sharma, 2018](https://arxiv.org/pdf/1801.01973) pointed out several weaknesses of this metric, including non-robustness under the Inception model's retraining and its non-transferrability to image datasets other than ImageNet. They also show how optimizing for IS (either in training, e.g. as an early stopping criterion or in model selection) promotes overfitting and the generation of adversarial examples for the Inception model. They thus suggest using IS as an informative metric, moreover, switch to $\log IS$ and fine tune or retrain it for the dataset where the generating model is trained.

#### **Frechet Inception Distance (FID)**

Introduced by [Heusel et al., 2017](https://papers.nips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf), FID also uses the Inception v3 model but without the final classification layer producing a 2048 dimensional activation vector. We then apply this both to the generated samples and the data distribution and fit a Gaussians on the resulting datapoints for each (fitting just means taking sample averages and computing empirical covariances) Finally the FID is calculated as the Frechet(Wasserstein-2)-distance of the resulting two normals:

$$ FID = d_F(\mathcal{N}(\mu_p, \Sigma_p), \mathcal{N}(\mu_q, \Sigma_q)), $$

where the Frechet distance of two probability distributions is defined as

$$ d_F(\nu_1, \nu_2) = \left(\inf_{\gamma\in\Gamma(\nu_1, \nu_2)}\mathbb{E}_{(x_1,x_2)\sim\gamma}\|x_1 - x_2\|^2\right)^{\frac{1}{2}}. $$

Here $\Gamma(\nu_1,\nu_2)$ is the set of probability distributions such that their marginals are $\nu_1$, and $\nu_2$. For Gaussians, this takes a simple form which translates to

$$ FID = \|\mu_p - \mu_q\|^2 + \mathrm{Tr}\left(\Sigma_p+\Sigma_q - 2(\Sigma_p\Sigma_q)^{\frac{1}{2}}\right)$$

Unfortunately FID, suffers from many of the same limitations of IS and care should be exercised when relying on it. For example, memorizing the training set will likely result in both very high FID and IS.

Finally, we mention that while strictly speaking FID only applies to image generation, [Unterthiner et. al, 2019](https://openreview.net/pdf?id=rylgEULtdN) proposed versions (FVD) for video generation by replacing Inception v3 with a network that considers the temporal coherence of the visual content across a sequence of frames as well. Similarly, [Kilgour et al., 2019](https://arxiv.org/pdf/1812.08466) proposed a variant (FAD) that is suitable for models generating audio.

#### **Precision and Recall**

A major shortcoming of the above single dimensional metrics is that they cannot distinguish between different failure modes. In particular, they don't distinguish between diversity/coverage and realism of the samples. Introduced in it's earlier form by [Lucic et al. (2028)](https://arxiv.org/pdf/1711.10337), *precision* and *recall* attempts to fill this gap. Intuitively, precision measures the quality of samples from $p_\theta$ while recall indicates the proportion of $q$ covered by $p_\theta$. [Sajjadi et al. (2018)](https://arxiv.org/pdf/1806.00035) proposed a computable notion of precision and recall

improved by [Kynkäänniemi et al. (2019)](https://arxiv.org/pdf/1904.06991)...

[TODO]

## III. Learning and generating from unconditioned distributions with diffusion models

After this long introduction, let us finally dive into diffusion models. In this section, we consider the unconditioned case first where we simply want to learn the data distribution without any further conditioning information. We start by giving an architecture agnostic introduction to the main ideas, namely the backward and forward diffusion processes. We then move on to score based generative modeling and show how the two seemingly different approaches are actually different perspectives of the same model family.

### The model and the backwards process

Starting with the backwards process is highly unusual in the long tradition of diffusion model expositions. We do it nevertheless because we believe that starting with the description of the model provides a better logical flow and motivates the role of the forward process better.

Diffusion models, proposed by [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585), are hierarchical latent variable models for the observed examples $x_0\in\mathbb{R}^D$ where the latent variable is a discrete time trajectory 

$$z=(x_1, \cdots, x_T)=:x_{1:T}\in(\mathbb{R}^D)^T.$$

As with other latent variable models, e.g. VAES, the model consist of a prior $p(x_{1:T})$ and a decoder $p(x_0\|x_{1:T})$. In the diffusion context the prior is given by a backwards Markov process with an isotropic Gaussian initial distribution $p(x_T)=\mathcal{N}(x_T; 0, I)$ and a parametrized transition function $p_{\theta}(x_t\|x_{t-1})$. The full latent distribution is then given by

$$ p_{\theta}(x_{1:T}) = p(x_T)\prod_{t=2}^T p_{\theta}(x_{t-1}|x_t)dx_{1:T}. $$

We also make the Markovian assumption to the decoder $p_\theta(x_0\|x_{1:T})=p_{\theta}(x_0\|x_1)$ making the marginal probability of the observed variable

$$ p_\theta(x_0) =\int p_{\theta}(x_0|x_{1:T}) p(x_{1:T})dx_{1:T}= \int p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)dx_{1:T}. $$

For large $T$, evaluating this high dimensional integral for training naively is hopeless with the exception of the simplest parametrizations of the transition function that allow for analytic computation (but greatly restrict the flexibility of the model). Monte-Carlo sampling comes to the rescue, but doing it naively from the prior would be of extremely high variance, most latent variable values are not compatible with the observed $x_0$ under $p_\theta$.

Instead, one can employ an importance sampling scheme using another distribution $q$ over the trajectories $x_{1:T}$, ideally one that emphasizes plausible latent values given $x_0$. Indeed, we know from VAEs that the ideal one would be the posterior of the latent $p_{\theta}(x_{1:T}\|x_0)$. We cannot compute this, however it gives us an idea that we should choose $q$ to be a process starting from the observed sample $x_0$. After making a
Markovian assumption on $q$ for tractability, we arrive at the importance distribution

$$ q(x_{1:T} | x_0) = \prod_{t=1}^Tq(x_t|x_{t-1}). $$

The model probability can then be written as

$$
p_{\theta}(x_0) = \int q(x_{1:T}|x_0)p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} dx_{1:T} = \mathbb{E}_{x_{1:T}\sim q(\cdot|x_0)}p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})},
$$

which can be readily evaluated by sampling trajectories from $q(.\|x_0)$ which was chosen to produce plausible latents thereby reduce the variance of the resulting estimator.

### The forward process

To complete the VAE analogy, $q(x_{1:T} \| x_0)$ plays the role of the encoder. Unlike VAE-s however, diffusion processes all but fix this distribution, also known as the forward process, since in this case the prior is the learnable one. The only exception from this is the seed of the backwards process which is set to be an isotropic Gaussian $p(x_T) =\mathcal{N}(x_T; 0, I)$ and accordingly, the forward process should be chosen in a way such that $q(x_T)$ is close to $p(x_T)$.

Luckily, a large class of Markov processes converge to their invariant (or stationary) distribution $\pi(x)$ satisfying

$$\int q(x_t|x_{t-1}) \pi(x_{t-1}) = \pi(x_t).$$

irrelevant of the initial value $x_0$. These Markov chains are called ergodic defined by the property $q(x_t\|x_0)\approx \pi(x_t)$ for large $t$. The seminal paper [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) proposed $q(x_{t}\|x_{t-1})$ to replace some amount of the original signal in $x_{t-1}$ with  Gaussian noise and in the end arrive at $x_T$ being close to the desired isotropic Gaussian. In particular,

$$ q(x_t|x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{1-\beta_t}x_{t-1}, \beta_t I), $$

where $\beta_t$ is called the diffusion step. We can also use the *reparametrization trick* to represent this as a difference equation

$$ {\bf x}_t = \sqrt{1-\beta_t}{\bf x}_{t-1} + \sqrt{\beta_t}\varepsilon_t, \qquad \varepsilon_t\sim\mathcal{N}(0, I)$$

The arbitrary looking choice of $\sqrt{1-\beta_t}$ makes the standard isotropic Gaussian a stationary distribution. Indeed, if $x_{t-1}$ is distributed as $\mathcal{N}(0, I)$ then $x_t$ is also a zero mean Gaussian with diagonal covariances $(1-\beta_t) + \beta_t = 1$.

#### **Convergence to Gaussian stationary distribution**

To take idea this further, let us borrow the notation from [Ho et al (2020)](https://arxiv.org/pdf/2006.11239):

$$ \alpha_t = 1-\beta_t,\qquad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s $$

which allows us to unfold the forward process to

$$ {\bf x}_t = \sqrt{\bar{\alpha}_t}{\bf x}_0 + \sqrt{1-\bar{\alpha}_t}\bar{\varepsilon}_t. $$

Here the $\bar{\varepsilon}_t$ are still isotropic standard Gaussians, but they are not independent anymore for different $t$. The reader is invited to prove this by induction.

For large $t$, as long as $\bar{\alpha}_t\to 0$ (which translates to the $\beta_t$-s not being too small), we have $\bf{x_t} \approx \bar{\varepsilon}_t \sim \mathcal{N}(0, I)$, which can be established rigorously by using e.g. [characteristic functions](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)). Furthermore, if the $\beta_t$ are increasing as in most practical implementations, then $\bar{\alpha}_t\leq (1-\beta_1)^t$ and the convergence is exponentially fast.

### The learning objective I: ELBO

Naturally, we would like the learned model $p_{\theta}(x_0)$ to approximate $q(x_0)$ in KL-divergence which we have seen to be equivalent to minimizing their cross-entropy $CE(q, p_{\theta})$:

$$
\begin{align*}
-\int q(x_0)\log p_\theta(x_0)dx_0 = - \mathbb{E}_q\log p_{\theta}(x_0) = -\mathbb{E}_{q(x_0)}\log \mathbb{E}_{x_{1:T}\sim q(\cdot|x_0)}\left[p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]
\end{align*}
$$

where the inner expectation is over the diffusion trajectories with fixed $x_0$ while the outer expectation is over $x_0$. If we wanted to evaluate this, we could sample $x_0$ from our dataset, simulate a $B$ trajectories $x_{1:T}^b$ from the forward process $q$ for each sample, and use Monte-Carlo evaluation to turn the expectation over averages:

$$ \hat{CE}(q, p_\theta) = -\frac{1}{N}\sum_{i=1}^N \log \frac{1}{B}\sum_{b=1}^B\left[p(x_T^{b})\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}^{b}|x_t^{b})}{q(x_t^b|x_{t-1}^b)}\right]$$

Unfortunately, this is potentially heavily biased estimator due to the logarithm (try computing the mean of the estimator, you will get stuck at $\mathbb{E}\log\sum_{samples}$).  

Instead, the VAE recipe suggests to use Jensen's inequality to swap the logarithm and the inner expectation and obtain the usual *evidence lower bound (ELBO)* ("lower" refers to the viewpoint of maximizing the likelihood where there are no minus signs and the inequality is flipped):

$$
-\int q(x_0)\log p_\theta(x_0)dx_0 \leq -\mathbb{E}_{x_{0:T}\sim q}\log \left[p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right] =: L
$$

where the expectation is now taken over the full trajectory $x_{0:T}$. Since Jensens was only applied to the inner expectation, equality will hold if and only if the term under the log is the same for all trajectories $x_{1:T}$. This would imply $q(x_{1:T}\|x_0)\propto p_\theta(x_{1:T}, x_0)$ which, after normalization, translates to $q(x_{1:T}\|x_0) = p_\theta(x_{1:T}\| x_0)$ that is when the distribution of the forward trajectory is exactly the posterior of $x_{1:T}$ under $p_{\theta}$ given $x_0$.

This exact posterior is, of course, no easier to compute than the model probability itself, and VAEs jointly optimize the encoder $q(z\|x)$ and the decoder $p(x\|z)$ to simultaneously close the Jensen-gap and the optimize the bound. For diffusion models, however, the distribution of the forward process is fairly rigidly determined with the $\beta_t$-s being the only (potentially) learnable *variational parameters*. The hope is that the trainable full prior will bring us closer to this fixed $q$ instead.

### The learning objective II: Sum of KL-divergences

At this point, one could provide the parametrization for $p_{\theta}$, evaluate with Monte-Carlo, take gradients and train as usual. This sampling, however is fairly expensive and it turns out we can do better. Let us first separate the edge terms at $t = 0$ and $t=T$ in $L$:

$$
L = -\mathrm{E}_q\log p(x_T) -\sum_{t=2}^T\mathrm{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} - \mathbb{E}_q\log\frac{p_\theta(x_0 | x_1)}{q(x_1|x_0)}
$$

Note that the only possible learnable parameters in the first term (which is a cross-entropy between $q(x_T\|x_0)$ and $p(x_T)$) are the $\beta_t$-s and can be ignored if we decide to treat $\beta_t$-s as hyperparameters instead.

[Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) eliminates the last term by pinning $p_\theta(x_0 \| x_1)$ to equal the reverse of $q(x_1\|x_0)$ under the stationary distribution of the forward process (which is an isotropic Gaussian):

$$
p_\theta(x_0 \| x_1) = q(x_1\|x_0)\frac{\pi(x_0)}{\pi(x_1)}.
$$

This is however a somewhat arbitrary choice as the forward chain is definitely not in stationary state at the start unless $q(x_0)=\pi(x_0)$ which never happens. Therefore, [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) decided to learn $p_{\theta}(x_0\|x_1)$ instead.

The key questions are what do we do with the terms in the middle sum and what parametric form should $p_\theta$ have. First note that these terms are almost KL-divergences but not quite because the probabilities in the numerator and the denumerator are over different variables. To rectify this, we can reverse the $q(x_t\| x_{t-1})$ term using Bayes's theorem:

$$
q(x_t|x_{t-1})=q(x_{t-1} | x_t) \frac{q(x_{t-1})}{q(x_{t})}
$$

If we wanted to evaluate this in practice, we would have to condition on $x_0$ and integrate over the intermediate variables. Since $q(x_0)$ is a very complicated distribution, the result will not be Gaussian and our only option would be to sample from the training set. On the other hand, a very neat trick proposed by [Sohl-Dickstein et al. (2015)](https://arxiv.org/pdf/1503.03585) is to realize $q(x_t\|x_{t-1}) = q(x_t\|x_{t-1}, x_0)$ by the Markov property and then applying Bayes gives

$$ q(x_t|x_{t-1}) = q(x_{t-1} | x_t, x_0) \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}. $$

What we gained is that now all terms are Gaussian. This is evident for all but $q(x_{t-1}\|x_t, x_0)$ which is then Gaussian by the virtue of this equation. Plugging back into the summands of the middle term in $L$ we get

$$ \mathbb{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} = \mathbb{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)} + \mathbb{E}_q \log\frac{q(x_{t-1}|x_0)}{q(x_{t}|x_0)}, $$

where from the second term, only the denominator for $t=T$ and the numeratorfor $t=2$ survive the subsequent summation. This leads to

$$
\begin{align*}
L = &-\mathbb{E}_q\log \frac{p(x_T)}{q(x_T|x_0)} - \sum_{t=2}^T\mathbb{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)} - \mathbb{E}_q\log p_\theta(x_0|x_1)
\end{align*}
$$
which can be readily written as KL divergences:

$$
\begin{align*}
&~ \mathbb{E}_qD_{KL}(q(x_T|x_0)\|p(x_T)) + \sum_{t=2}^T \mathbb{E}_qD_{KL}(q(x_{t-1}|x_t, x_0)\| p_{\theta}(x_{t-1}|x_t)) + \mathbb{E}_qCE(p_{\theta}(x_0|x_1)|| q(x_0, x_1) )
\end{align*}
$$

Let us use the abbreviation $L_T + \sum_{t=1}^{T-1} L_t + L_0$. Then $L_T$ is the KL-divergence between two Gaussians and is usually exponentially small for large $T$ due to the ergodicity of $q(x_t\|x_0)$. Again, if the $\beta_t$-s are not trainable then this term has no trainable component and can be dropped.

### Parametrization of the backwards model

$L_t$ is now the KL-divergence between a Gaussian variable and $p_{\theta}(x_{t-1}\| x_t)$. This suggests choosing $p_{\theta}$ to be a Gaussian with mean and variance that are learnable functions of $x_t$:

$$ p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t)) \qquad 1<t\leq T.$$

This makes life particularly nice, because the KL-divergence between two Gaussians can be computed in closed form in terms of their parameters. All we need then is to compute the mean and variance of $q(x_{t-1}\|x_t, x_0)$:

$$
\begin{align*}
-2\log q(x_{t-1}|x_t, x_0) =& -2\log q(x_t|x_{t-1}, x_0) - 2\log q(x_{t-1}|x_0) + 2\log q(x_t|x_0) = \\
& C(x_0, x_t) + \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{\beta_t} + \frac{(x_{t-1}-\sqrt{\bar{\alpha}}_tx_0)^2}{1-\bar{\alpha}_t} = \\
& C(x_0, x_t) + \left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_t}\right)x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\bar{\alpha}_t}{1-\bar{\alpha_t}}x_0\right)x_{t-1} = \\
& C(x_0, x_t) + \tilde\beta^{-1}(x_{t-1} - \tilde\mu(x_t, x_0))^2
\end{align*}
$$
where $C(x_0, x_t)$ denotes a quantity (possibly different each line) that only depends on $x_0$ and $x_t$ but not on $x_{t-1}$ and
$$
\tilde\beta_t = \left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_t}\right)^{-1} = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
$$
$$
\tilde\mu(x_t, x_0) = \tilde\beta_t\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\bar{\alpha}_t}{1-\bar{\alpha_t}}x_0\right) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0.
$$
In the formula for the mean, we can eliminate $x_0$ by utilizing the reparametrization trick representation $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar\alpha_t}\bar\varepsilon_t$ to get

$$\tilde\mu(x_t, \bar\varepsilon_t) = \tilde\mu(x_t, x_0(x_t, \bar\varepsilon_t)) =  \frac{1}{\sqrt{\alpha_t}}\left(x_t -\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\bar\varepsilon_t\right)$$

Putting it together, $q(x_{t-1}\|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde\mu(x_t, \bar\varepsilon_t), \tilde{\beta} I)$.

### The learning objective III: Final form

To compute the KL-divergence of two normals, we can use the well known formula

$$ D_{KL}(\mathcal{N}_1, \mathcal{N}_2) = \frac{1}{2}\left[\mathrm{Tr}(\Sigma_2^{-1}\Sigma_1) - D + (\mu_2 -\mu_1)\Sigma_2^{-1}(\mu_2 -\mu_1) + \log\frac{\mathrm{det}\Sigma_1}{\mathrm{det}\Sigma_2}\right]$$

where $D$ is the dimension of $x to arrive at

$$ L_t = \frac{1}{2}\left[\tilde{\beta}_t\mathrm{Tr}\Sigma^{-1}_\theta(x_t, t) - D + (\mu_{\theta}(x_t, t)-\tilde\mu(x_t, x_0))^T\Sigma_\theta^{-1}(x_t, t) (\mu_{\theta}(x_t, t)-\tilde\mu(x_t, x_0)) + D\log\tilde\beta_t - \log\det\Sigma_{\theta}(x_t, t)\right]. $$

Since the covariance matrix of the forward transition is diagonal, it is reasonable to postulate $\Sigma_\theta(x_t, t) = diag(\Sigma_{\theta}^{ii}(x_t, t))$, which simplifies the formula to a sum of one dimensional KL-divergences

$$ L_t = \frac{1}{2}\mathbb{E}_{x_0, x_t \sim q}\sum_i\left[ \frac{\tilde{\beta}_t + (\mu_{\theta}^i(x_t, t)-\tilde\mu^i(x_t, x_0))^2}{\Sigma_{\theta}^{ii}(x_t, t)} - \sum_i\log\Sigma_{\theta}^{ii}(x_t, t) - 1\right] + \frac{D(\log\tilde\beta_t -1)}{2}$$

This is where [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) effectively puts down the pen. They set $(\mu_{\theta}(x_t, t), \sigma_{\theta}(x_t, t))$ to be a Neural Network (see below for architectures) and learn them together $\beta_t$ by gradient descent. On the other hand, [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) set $\Sigma^{ii}_{\theta}(x_t, t)$ to be a constant $\sigma_t^2$ and treat the $\beta_t$-s (and thus the $\tilde \beta_t$-s) as hyperparameters. After dropping the now non-trainable terms, this approach ends up with 

$$ L_t = \frac{1}{2\sigma_t^2}\mathbb{E}_{x_0, \bar\varepsilon_t}\|\tilde\mu(x_t, \bar\varepsilon_t) - \mu_{\theta}(x_t, t)\|^2 $$

that we could directly optimize, however, we can go even further by rewriting the predicted mean as

$$
\mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha}_t}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_{\theta}(x_t, t)\right).
$$

This is the definition of $\varepsilon_\theta$ so that it matches the formula we have for $\tilde\mu$. With this, $L_t$ above becomes
$$
L_t = \frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\sigma_t^2}\mathbb{E}_{x_0, \bar\varepsilon_t}\|\bar\varepsilon_t - \epsilon_{\theta}(x_t,t)\|^2 = \frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\sigma_t^2}\mathbb{E}_{x_0, \bar\varepsilon_t}\|\bar\varepsilon_t - \epsilon_{\theta}(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\bar\varepsilon_t,t)\|^2
$$

Other than the weighting factor before the expectation, this says that we want to use a neural network to learn $\varepsilon_\theta$ which takes a noisy image as input and an index of what stage of the denoising we are in and predicts the noise that was added to the original image.

We mention that [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) reported better training result by dropping the time dependent pre-factor and using

$$ L_{simple, t}(\theta) = \mathbb{E}_{x_0, \bar\varepsilon_t}\|\bar\varepsilon_t - \epsilon_{\theta}(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\bar\varepsilon_t,t)\|^2. $$

This corresponds to the ELBO bound being replaced by a weighted version that emphasizes loss terms corresponding to larger $t$. The authors speculate that this helps because the amount of noise injected is commonly smaller in the early steps so denoising these is an easier task that we do not need to enforce as strong.

### The decoder

[Ho et al (2020)](https://arxiv.org/pdf/2006.11239) derives a discrete decoder $p_{\theta}(x_0\|x_1)$ to obtain valid images assuming that all image data consists of integers in ${0, 1, \dots, 255}$ linearly scaled to $[-1, 1]$.

$$
p_{\theta}(x_0|x_1) = \prod_{i=1}^D\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}(x; \mu_{\theta}^i(x_1, 1), \sigma_1^2)dx
$$

$$
\delta_{\pm}(x) = \left\{\begin{array}{cc}
\pm\infty&\textrm{if}~x = \pm 1\\
x\pm\frac{1}{255}&\textrm{if}~\pm x < 1
\end{array} \right.
$$
The effect of this is simply to discretize the probability mass to multiples of $1/255$ to define actual images.

### The $\beta_t$ noise schedule

The $\beta_t$-s are a very important set of parameters of a diffusion model that controls how fast the noise is injected by the forward process. [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) proposed to learn them, while [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) treated them as hyperparameter, increasing linearly from $\beta_1=10^{-4}$ to $\beta_T=0.02$ remaining relatively small compared to normalize image pixel values.

[Nichol & Dariwal (2021)](https://arxiv.org/pdf/2102.09672) argued that the linear schedule is too eager to destroy the data and proposed a slower schedule while keeping the property of $\alpha_t$ having a linear drop in the middle of the range and being flat at both ends. Namely,

$$ \alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos\left(\frac{t/T +s}{1+s}\frac{\pi}{2}\right)^2 $$

and then $\beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}$ clipping it to be less then $1-\delta$ for some very small $\delta$. This is done to avoid total destruction at $t=T$ and similarly, the offset $s$ is introduced so that $\beta_t$ is not too small.

The number of steps should be chosen to make $L_T$ negligibly small. In practice, $T$ being a couple of thousand has been a standard choice.

### The reverse process variance $\Sigma_\theta$

There are multiple possible options here as well, [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) proposed learning a diagonal but otherwise unconstrained covariance matrix. [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) empirically argued that this leads to poorer sample quality and unstable training due to the small reasonable range for its magnitude in $[\beta_t, \tilde\beta_t]$. Instead, they decided to fix it to $\sigma_t^2I$ where $\sigma_t$ is either set to $\beta_t$ or $\tilde{\beta}_t$ reporting similar performance for both choices.

[Nichol & Dariwal (2021)](https://arxiv.org/pdf/2102.09672) showed that $\beta_t$ and $\tilde\beta_t$ are very similar with the exception of small $t$-s that happen to also contribute the most to the overall loss $L_{simple}=\sum_{t=0}^TL_{simple, t}$. They thus propose to return to learning $\Sigma_\theta(x_t, t)$ but with a special parametratization to avoid the training instability. Namely, the model learns an extra vector $v\in\mathbb{R}^D$ and use it to interpolate between $\beta_t$ and $\tilde\beta_t$ componentwise:

$$ \Sigma_\theta(x_t,t) = e^{v\log\beta_t + (1-v)\log\tilde\beta} $$

Since $L_{simple}$ contains no signal for the variances, they proposed a hybrid objective

$$L_{hybrid, t} = L_{simple, t} + \lambda L_t, $$

where $L_t$ is the original unweighted ELBO loss with a stop gradient on the $\mu$ output.

Nevertheless, they still found the gradients of $L_t$ to be very noisy and in light of the differences in $L_t$ magnitudes for different $t$, proposed a modification to the selection of $t$ in the training algorithm above. Namely, they proposed to sample $t$ with an importance sampling scheme:

$$\hat{L}_t = \mathbb{E}_{t\sim p_t}\left[\frac{L_t}{p_t}\right] $$

where $p_t\propto\sqrt{\mathrm{E}L_t}$ is estimated by a moving average of $L_t$ with an initial uniform selection phase until we obtain enough sample for each (this is akin to a multi-armed bandit).


### Training, Inference, Sampling

#### **Training**

Note that we share the neural network $\varepsilon_\theta, it is sufficient for each training step to sample a single timestep. The training loop thus consists of the following steps:

1. Choose a random element $x_0$ of the training set and sample an integer $t$ uniformly in $\{1,\dots, T\}$
2. Sample $\varepsilon\sim \mathcal{N}(0,I)$.
3. Take gradient descent step on $\nabla_\theta \\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon)\\|^2$.

#### **Inference**

To evaluate the model probability of an unseen example, I believe we still have to do Monte-Carlo sampling to approximate

$$ p(x_0) = \mathbb{E}_{x_{1:T}\sim q(\cdot|x_0)}p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}. $$

and is consequently fairly expensive consisting of the generation of $B$ trajectories and have $B\cdot T$ neural network plus functions-on-top evaluations.

1. Generate $B$ simulated trajectories from the forward process.
2. For each $b = 1,...,B$, and each $t\in 1, \dots T$, evaluate

    (a) $\hat\varepsilon_t^{(B)} = \varepsilon_\theta(x_t^{(B)}, t)$

    (b) transform it into $\hat{\mu}_t^{(B)}= \mu _{\theta}(x_t^{(B)}, \hat \varepsilon_t^{(B)})$

    (c) evaluate $p_{\theta}(x_{t-1}^{(B)}\|x_t^{(B)})=\mathcal{N}(x_{t-1}^{(B)}; \hat{\mu}_t^{(B)}, \sigma_t^2 I)$ using the Gaussian pdf.
3. For each $b = 1, ..., B$, compute $p_B(x_0)$ by evaluating Gaussians to get the rest of the integrand. The final step involves the discretization step.
4. Output $p(x_0) = \frac{1}{B}\sum_{b=1}^Bp_B(x_0)$

#### **Sampling**

Sampling is ever so slightly simpler as we only need to go through a single backwards pass and have only $T$ neural network plus functions-on-top evaluations.

1. Generate $x_T\sim\mathcal{N}(0, I)$.
2. For each $t = T,\dots, 2$, sample $z\sim\mathcal{N}(0, I)$ and update

$$ x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_{\theta}(x_t, t)\right) + \sigma_t z $$

3. Output $\mu_{\theta}(x_1, 1)$.

### Summary

This was a lot, let us recap. Diffusion models are latent variable models, where the latent variable is given by the trajectory of a backwards Markov Chain $z = x_{T:1}$. Unlike VAE-s, only the distribution of $x_T$ is specified by hand to be an isotropic Gaussian, the rest of the distribution come from a learnable Markov process. To learn this chain by the negative log-likelihood, we do importance sampling with the proposal distribution given by a very simple and tracktable Markov chain starting from $x_0$ and use the ELBO inequality to obtain a tracktable optimization objective. This bound then decomposes into boundary terms $L_T$, $L_0$ and internal terms $L_1,\dots, L_{T-1}$. The internal terms are all KL-divergences between the posterior of the forward transition and the backward transition. If the learnable model has Gaussian transitions, then we can explicitly write this KL-divergence down in terms of the means and variances (after assuming a diagonal covariance structure). If the covariances are isotropic, this will imply an $L^2$ loss on the predicted mean, what's more, suprisingly, we can take this even further when the learning problem becomes learning the noise in the noisy image under $L^2$ loss.

## IV. Score based generative models

While score based learning at first glance is unrelated to diffusion models (and was developed independently), later research uncovered that the connection between the two are much stronger than previously believed. To help illustrate this, we first cover.

### Score matching

Often identify or estimate the distribution of the date with a statistical model only proportionally $p_{\theta}(x)\propto\tilde{p}_{\theta}(x)$. This means that we get the dependence on $x$ right (up to estimation), however, this is not enough for likelihood based models that require knowledge of the normalization constant (also known as the partition function) making

$$ p_{\theta}(x) = \frac{\tilde p_\theta(x)}{Z_{\theta}}. $$

integrate to one. Clearly, $Z_{\theta} = \int \tilde{p}_\theta(x)dx$, but this integral is often times completely intractable when $x$ is high dimensional even by numerical integration. Approximations can be made but they are often poor and Markov Chain Monte Carlo (MCMC) is either too high variance or impractically slow to be used in maximizing the log-likelihood 

$$ \mathbb{E}_q\log p_{\theta}(x) = \mathbb{E}_q \log \tilde p_\theta(x) - \log Z_\theta $$

where $q$ is again the target distribution the observed data comes from.

[Hyvarinen (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) demonstrates that naively trying to optimize the unnormalized first term leads to flat probabilities as one way to assign high unnormalized likelihood to the data, when not constrained by normalization, is to assign it to every $x$. Instead the author proposes looking at the Stein score functions, which is the gradient of the log probability and does not depend on the normalization constant:

$$ s_q(x) = \nabla_x\log q(x),\qquad s_{p}(x;\theta) = \nabla_x\log p_{\theta}(x) = \nabla_x\log\tilde p_\theta(x). $$

The model can then be trained by tuning the $\theta$ such that the model and data scores are close. The proposed optimization objective is the Fisher divergence ([LINK TO ABOVE])

$$ J(\theta) = \frac{1}{2}\mathbb{E}_q\|s_p(x; \theta) - s_q(x)\|^2. $$

This is unfortunately not immediately computable since we don't have access to $s_q(x)$ directly. There are multiple solutions here, we look at an exact option now and address the option of adding noise to the data in the next section. 

Remarkably, it turns out that a simple integration by parts trick shows that the Fisher divergence is equivalent to

$$ J(\theta) = \mathbb{E}_q\left[\mathrm{Tr}(\nabla_x s_p(x; \theta))+\frac{1}{2}\|s_p(x;\theta)\|^2\right], $$

which does not involve the score of the data distribution in the expectand and hence can be empirically estimated by sampling from the data. The trade-off is that now have to be able to compute the trace of the Jacobian of the model score (or the Hessian of the log-probability) in the first term. For neural network based models, this can be done through automatic differentiation, but it requires $D$ extra backwards passes.

For training with access to the unnormalized model probabilies, we have to do $D$ forward/backwards passes in $x$ to compute the scores, $D$ more to compute the trace, and then we need a backwards pass with respect to $\theta$ as well. This approach thus only allowed for simple models or for low dimensional problems and was not suitable at scale.

#### **Spliced score matching**

To alleviate this computational burden, [Song et al. (2019a)](https://arxiv.org/abs/1905.07088) proposed to replace the full score comparison with comparison of random projections of the score. That is, the objective function is replaced by

$$ J(\theta) = \frac{1}{2}\mathbb{E}_{v\in p_v}\mathbb{E}_q\|v^Ts_p(x; \theta) - v^Ts_q(x)\|^2. $$

where $p_v$ is a simple $D$ dimensional distribution, most likely an isotropic Gaussian or a Rademacher distribution. The same integration by parts trick now leads to

$$ J(\theta) = \mathbb{E}_{v\sim p_v}\mathbb{E}_q\left[v^T\nabla_x s_p(x; \theta)v+\frac{1}{2}(v^T s_p(x;\theta))^2\right]. $$

When using Gaussian or Rademacher, the expectation of the second term under $p_v$ can be computed explicitly leading to the reduced variance objective

$$ J(\theta) = \mathbb{E}_{v\sim p_v}\mathbb{E}_q\left[v^T\nabla_x s_p(x; \theta)v+\frac{1}{2}\|s_p(x;\theta)\|^2\right]. $$

#### **Denoising score matching**

Another approach starting from the original Fisher divergence was put forward by [Vincent (2011)](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) inspired by Denoising Autoencoders where the autoencoder has to reconstruct the original input from a noise-perturbed one. He suggested to perturb the empirical data distribution by some Gaussian noise that is the target distribution for a datapoint $x$ gets spread out according to

$$ q_\sigma(\tilde x | x) = \mathcal{N}(\tilde x; x, \sigma^2 I). $$

The joint distribution is given by $ q_\sigma(x,\tilde x) = q_{\sigma}(\tilde x \| x) q(x)$ and the score matching objective becomes

$$ \frac{1}{2}\mathbb{E}_{(x,\tilde x)\sim q_{\sigma}(x, \tilde x)}\|s_p(\tilde x, \theta) - \nabla_{\tilde x}\log q(\tilde x | x)\|^2. $$

That is, the score model is learning the perturbed distribution. In exchange, however, we can compute the target score since

$$ \nabla_{\tilde x}\log q(\tilde x | x) = \frac{x - \tilde x}{\sigma^2}$$

which points towards the direction of the clean data.

### Sampling with Langevin Dynamics

No matter how we learned the score model, we want to use it to genrate samples from it. Luckily, inspired by 19th century physics, [Welling & Teh, 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) proposed just the right tool

$$ \tilde x_t = \tilde x_{t-1} + \frac{\delta}{2}\nabla_x\log p_\theta(\tilde x_{t-1}) + \sqrt{\delta}\varepsilon_t,\qquad \varepsilon_t\sim\mathcal{N}(0, I) $$

This problem can be thought of as a noise perturbed gradient descent on the log probability and clearly it only needs information on the model score. Note that $x_{t\delta} = \tilde x_{t}$ is a simple Euler-Maruyama discretization scheme of the Stochastic Differential Equation (SDE)

$$ dx_t = \frac{1}{2}\nabla_x\log p_\theta(x_t) + dw_t $$

with [has stationary distribution](https://math.stackexchange.com/questions/4200043/langevin-equation-and-convergence-to-stationary-solutions-free-energy-sde-fpe) $p_{\theta}(x)$ to which $p(x_t)$ converges to as $t\to\infty$. This means that if $\delta$ is small and $T$ is large, then $\tilde x_T$ will be approximately distributed according to $p_\theta(x)$ which we can leverage to generate samples. For finite $\delta$ and $T$ we could apply Metropolis-Hastings corrections, but we will not pursue this here.

[Song et al. (2019b)](https://arxiv.org/pdf/1907.05600) pointed out two serious challenges. We have already addressed the first which is the difficulty in gradient estimation in the ambient space when the data lives on a lower dimensional manifold and discussed how denoising score matching can help. The second one is that low data density regions (where a randomly initialized process will most likely start) both impair score estimation and slow down the mixing of Langevin dynamics as the process will get trapped in the modes of the distribution with only rare transitions in between.

### Noise condiditonal score networks (NCSN)

To mitigate these problems, [Song et al. (2019b)](https://arxiv.org/pdf/1907.05600) proposed an annealing Langevin process where choose $L$ noise levels $\sigma_1 > \dots > \sigma_L$ and at each level perturbs the data with a corresponding isotropic Gaussian as in denoising score matching. They then put forward training a joint network $s_p(x, t;\theta)$ by minimizing the objective

$$ \frac{1}{L}\sum_{i=1}^L\lambda(\sigma_i) l(\theta,\sigma_i), $$

where noise level loss $l(\theta,\sigma_i)$ is defined by

$$
\frac{1}{2}\mathbb{E}_{x\sim q}\mathbb{E}_{\tilde x\sim\mathcal{N}(x, \sigma_i)}\left\|s_{\theta}(\tilde x, \sigma_i)+ \frac{\tilde x -x}{\sigma^2} \right\|^2
$$

The function $\lambda(.)$ is chosen so that the different terms are roughly of the same magnitude. [Song et al. (2019b)](https://arxiv.org/pdf/1907.05600) reported finding $\lambda(\sigma) = \sigma^2$ to be a good choice.

To do inference from this model, the Langevin sampling will be done in an annealed fashion chaining together Langevin steps with decreasing noise size. At noise size $\sigma_i$, we will use the step size $\delta_i=\delta\frac{\sigma_i^2}{\sigma_L^2}$ where $\delta$ is the stepsize for the lowest noise level. This choice was found to be the right one to balance the Signal to Noise Ratio (SNR) given by the ratio of the deterministic and the noise term in the process. Finally, the proposed choice for the noise levels is that they follow a geometric progression $\sigma_i = \sigma_1 c^i$ for some $c\in (0, 1)$.

Finally, we mention that this initial setup worked well for 32x32 images but fell apart for larger resolution. Several improvements were proposed by [Song & Ermon (2020)](https://arxiv.org/pdf/2006.09011) unlocking scalability to higher resolution images that we do not detail here.

### Connection to diffusion models

There is a striking similarity between how both NCSN and Diffusion models use a hierarchical scale of adding more and more noise to the original data and then generate using a Markov process starting from noise. To see an even deeper connection, we proceed as follows. If we make the identification between the noise level $i$ and the diffusion step $t$, the score of the data produced by the forward process is

$$
s_q(x_t, t)\approx \nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t-\sqrt{\alpha_t}x_0}{1-\bar\alpha_t} = -\frac{\bar\varepsilon_t}{\sqrt{1-\bar\alpha_t}},
$$

where we used that $q(x_t|x_0)$ is a Gaussian and the definition of $\bar\varepsilon_t$. With a simple rescaling of our score model, we also have

$$ s_p(x_t, t; \theta) = - \frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\alpha_t}}, $$

and our diffusion model objective is equivalent to score matching at each step of the diffusion.

### Deeper connection with Diffusions: Discretization of an SDE

[Song et al. (2021)](https://openreview.net/pdf?id=PxTIG12RRHS) ...

[TODO]

## V. Conditioned generation

While generating images is already lots of fun, the real magic begins when we can guide the generation by a text prompt or other conditioning information.

[TODO]

## VI. Speeding up generation

[Song et al. (2022)](https://arxiv.org/pdf/2010.02502) then

[Nichol & Dhariwal (2021a)]


## VII. Latent Diffusion models

## VIII. Architectures

The presentation so far was architecture agnostic focusing on the learning paradigm, however, the architectures used are crucial to the performance of these models. In this section we close this gap by discussing some of the architectural choices that were made in the literature.

[TODO]

## Resources

Besides the references in the text, I took inspiration from [Lilian Wang's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process), the Stanford [Deep Generative Models course](https://deepgenerativemodels.github.io/), and [Yang Song's blogpost](https://yang-song.net/blog/2021/score/)