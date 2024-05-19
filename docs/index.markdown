---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

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

    - **Autoregressive models** (), e.g. most LLM-s, that factorizes the probability into the product of conditional probabilities according to some sometimes natural, sometimes arbitrary ordering: $p(x)=\prod_ip(x_i\|x_{<i})$. The model then learns these conditional distributions (and some seeding distribution for $x_0$). This structure makes the evaluation of the likelihood easy, but the generation has to be sequential which can be very slow.
    - **Variational Autoencoders (VAE)**, are latent variable models with a training methodology that jointly learns (1) what latent values are plausible for a given observation (encoder) and (2) how to decode a given value of the latent into a distribution over possible data domain instances. It does this to simultaneously optimize a surrogate loss function and close the gap between this surrogate and the likelihood.
    - **Normalizing flow models** that learn invertible deterministic mappings between the latent space and the observation domain and uses it to transform the latent densities to densities in the data domain. Needless to say, the invertibility puts a fair amount of restriction of the architecture of this mapping has been restricted to continuous distributions.
    - **Energy based models** [TODO]
    - **Diffusion models** that are latent variable models, however, the latent is the path of a diffusion process and the prior latent distribution is both learnable. They can also be viewed as stochastic flows or stacked VAE-s and use a similar surrogate objective approach. They are the subject of this survey and we will go into the topic in much more detail below.

2. **Implicit generative models** where the probability distribution is implicitly represented by its sampling process but without a tracktable density/mass function. They often require adversarial training which is notoriously unstable and often leads to model collapse.

   - **Generative Adversarial Networks (GAN)**, the state of the art for quite some time, where a generator model is trained to fool a discriminator model which is trained to tell generated and real images apart. They have excellent generation properties but are notoriously hard to train.

3. **Score based generative models** that lets go of tracking the normalization constant and instead models the gradient of the log-likelihood often referred to as the Stein score. They can be directly estimated by score-networks through a procedure known as score matching after which generation is possible using Langevin dynamics. They achieved state of the art performance at the time on many tasks. The drawback is that when according to the manifold hypothesis the ambient space is very high dimensional but the data lies on a much much smaller manifold, taking gradients in the ambient space can become somewhat ill defined.

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

For deep neural network models, the computation of the trace is fairly prohibited since all $D$ diagonal entries have to be computed through backpropagation. [Song et al 2019a](https://arxiv.org/pdf/1905.07088) proposed to mitigate this by comparing random, one dimensional projections of the scores which leads to the *sliced score matching* matching objective:

$$ J(\theta) = \mathbb{E}_{v\sim p_v}\mathbb{E}_q\left[v^T\nabla_x s_p(x; \theta)v+\frac{1}{2}(v^T s_p(x;\theta))^2\right], $$

where $p_v$ is a Rademacher (uniform on the vertices of the $D$ dimensional hypercube $\{\pm 1\}^D$) or an isotropic stadard normal.

#### **Inception Score (IS)**

IS, introduced by [Salimans et al., 2016](https://arxiv.org/pdf/1606.03498) to evaluate GAN-s, is a metric that evaluates generation by how well the generated images match the real images in the dataset in terms of their diversity and quality using statistic of the popular Inception v3 deep convolutional network designed for classification tasks on ImageNet. For a given image, it produces a vector of conditional probabilities $p_i(y\|x)$ over the thousand Image labels.

IS is then defined as the exponential of the expected KL divergence between the conditional and the marginal distribution under the generative distribution:

$$ IS = \exp(\mathbb{E}_{x\sim p_{\theta}}D_{KL}(p_{i}(y|x)\|p_i(y))),$$

where $p_i(y) = \mathbb{E}_{x\sim p_\theta}p_i(y\|x)$ is the marginal class distribution. To see what this metric is trying to accomplish, note that some easy manipulation shows

$$ \log IS = I(y;x) = H(y)-H(y|x),$$

that is, the logarithm of the IS is the mutual information between the class distribution and the generated sample. By the decomposition in the final step, a high IS corresponds to high entropy for the class distribution encouraging diversity, and low entropy for the Inception model's output encouraging clear images with a single object.

[Barratt & Sharma, 2018](https://arxiv.org/pdf/1801.01973) pointed out several weaknesses of this metric, including non-robustness under the Inception model's retraining and its non-transferrability to image datasets other than ImageNet. They also show how optimizing for IS (either in training, e.g. as an early stopping criterion or in model selection) promotes overfitting and the generation of adversarial examples for the Inception model. They thus suggest using IS as an informative metric, moreover, switch to $\log IS$ and fine tune or retrain it for the dataset where the generating model is trained.

#### Frechet Inception Distance (FID)

Introduced by [Heusel et al., 2017](https://papers.nips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf), FID also uses the Inception v3 model but without the final classification layer producing a 2048 dimensional activation vector. We then apply this both to the generated samples and the data distribution and fit a Gaussians on the resulting datapoints for each (fitting just means taking sample averages and computing empirical covariances) Finally the FID is calculated as the Frechet(Wasserstein-2)-distance of the resulting two normals:

$$ FID = d_F(\mathcal{N}(\mu_p, \Sigma_p), \mathcal{N}(\mu_q, \Sigma_q)), $$

where the Frechet distance of two probability distributions is defined as

$$ d_F(\nu_1, \nu_2) = \left(\inf_{\gamma\in\Gamma(\nu_1, \nu_2)}\mathbb{E}_{(x_1,x_2)\sim\gamma}\|x_1 - x_2\|^2\right)^{\frac{1}{2}}. $$

Here $\Gamma(\nu_1,\nu_2)$ is the set of probability distributions such that their marginals are $\nu_1$, and $\nu_2$. For Gaussians, this takes a simple form which translates to

$$ FID = \|\mu_p - \mu_q\|^2 + \mathrm{Tr}\left(\Sigma_p+\Sigma_q - 2(\Sigma_p\Sigma_q)^{\frac{1}{2}}\right)$$

Unfortunately FID, suffers from many of the same limitations of IS and care should be exercised when relying on it. For example, memorizing the training set will likely result in both very high FID and IS.

Finally, we mention that while strictly speaking FID only applies to image generation, [Unterthiner et. al, 2019](https://openreview.net/pdf?id=rylgEULtdN) proposed versions (FVD) for video generation by replacing Inception v3 with a network that considers the temporal coherence of the visual content across a sequence of frames as well. Similarly, [Kilgour et al., 2019](https://arxiv.org/pdf/1812.08466) proposed a variant (FAD) that is suitable for models generating audio.

[Other Generation metrics: TODO]

## III. Unconditioned generation with Diffusions

After this long introduction, let us finally dive into diffusion models.

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

### (b) The backward process

#### **The model probability**

It will be helfpul to think in analogy with VAE-s where the diffusion trajectory $z = x_{1:T}$ is for all intents and purposes a complex hidden variable. Then $q(x_{1:T} \| x_0)$ plays the role of the encoder and we are looking to learn both the full prior latent distribution $p_\theta(x_{1:T})$ and the decoder $p_{\theta}(x_0\|x_{1:T}). In what follows, we will use the VAE analogy often. The full prior distribution of the latent is obtained by takinga fixed $p(x_T)$ to be an isotropic standard multivariate Gaussian and using a learned Markov process working backwards on trajectory. In particular, we start with a Gaussian $p(x_T) \sim \mathcal{N}(x_T; 0, I)$ at time $T$ and proceed backward according to a parametrized transition kernel $p_{\theta}(x_{t-1} | x_t)$ that we will learn. With this, the decoder becomes $p_{\theta}(x_0|x_{1:T})= p_{\theta}(x_0|x_1)$.

The model probability can then be computed as

$$ p_\theta(x_0) =\int p_{\theta}(x_0|x_{1:T}) p(x_{1:T})dx_{1:T}= \int p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)dx_{1:T}. $$

In the latent variable language, this corresponds to $p(x_0) = \int p(x_0, z)dz$, computing the model probability by enumerating all possible latents that could produced $x_0$. As a result, this high dimensional integral is usually impossibly expensive to even approximate directly. The reader's thoughts might drift to do Monte Carlo sampling starting with the Gaussian $p(x_T)$, however, that leads to a very high variance estimator. This is because, most latents $x_{1:T}$ don't make sense for $x_0$ and will have very low conditional probability while we have a very high chance of missing high conditional probability latents. In short, the prior is great for sampling but not good for evaluation in training.

The VAE analogy comes to the rescue by guiding us towards utilizing the forward trajectory in an importance sampling scheme to emphasize plausible latent values given $x_0$. In other words, the latent proposal distribution is chosen to be $q(x_{1:T}\|x_0) = \prod_{i=1}^Tq(x_t\|x_{t-1})$, that is, the law of the forward trajectory launched from the starting point $x_0. With this,

$$
p(x_0) = \int q(x_{1:T}|x_0)p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} dx_{1:T} = \mathbb{E}_{x_{1:T}\sim q(\cdot|x_0)}p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})},
$$

which we can evaluate by simple Monte-Carlo by sampling by simulating the forward process. We are, however, not done yet, we haven't even use the fact that the model probability $p(x_0)$ should resemble the data distribution $q(x_0)$.

### **The learning objective**

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

This exact posterior is, of course, no easier to compute than the model probability itself, and VAEs try to jointly optimize the encoder $q(z\|x)$ and the decoder $p(x\|z)$ to simultaneously close the Jensen-gap and the optimize the bound. In our situation, however, the distribution of the forward process is fairly rigidly prescribed with the $\beta_t$-s being the only (potentially) learnable *variational parameters*. The hope is that the trainable full prior will bring us closer to this fixed $q$ instead.

At this point one could provide the parametrization for $p_{\theta}$, evaluate with Monte-Carlo, take gradients and train as usual. This sampling, however is fairly expensive and it turns out we can do better. Let us first separate the edge terms at $t = 0$ and $t=T$ in $L$:

$$
L = -\mathrm{E}_q\log p(x_T) -\sum_{t=2}^T\mathrm{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} - \mathbb{E}_q\log\frac{p_\theta(x_0 | x_1)}{q(x_1|x_0)}
$$

Note that the only possible learnable parameters in the first term (which is a cross-entropy between $q(x_T\|x_0)$ and $p(x_T)$) are the $\beta_t$-s. On the other end, [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) eliminates the last term by pinning $p_\theta(x_0 \| x_1)$ to equal the reverse of $q(x_1\|x_0)$ under the stationary distribution of the forward process (which is an isotropic Gaussian):

$$
p_\theta(x_0 \| x_1) = q(x_1\|x_0)\frac{\pi(x_0)}{\pi(x_1)}.
$$

This is however a somewhat arbitrary choice as the forward chain is definitely not in stationary state at the start unless $q(x_0)=\pi(x_0)$ which never happens. Therefore, [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) decided to learn $p_{\theta}(x_0\|x_1)$ separately.

The key element though is what we do with the terms in the middle sum and what parametric form should $p_\theta$ have. First note that these terms are almost KL-divergences but not quite because the probabilities in the numerator and the denumerator are over different variables. To rectify this, we should reverse the $q(x_t\| x_{t-1})$ term using Bayes's theorem:

$$
q(x_t|x_{t-1})=q(x_{t-1} | x_t) \frac{q(x_{t-1})}{q(x_{t})}
$$

Note, however, that the way to compute this is to condition on $x_0$ and integrate over the intermediate variables. Since $q(x_0)$ is a very complicated distribution, the result will not be Gaussian and we would have to sample from the training set at best. Instead, a very neat trick is to use the Markov property to realize $q(x_t\|x_{t-1}) = q(x_t\|x_{t-1}, x_0)$ and then apply Bayes theorem to this conditioned version,

$$ q(x_t|x_{t-1}) = q(x_{t-1} | x_t, x_0) \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}. $$

The gain is that now everything is Gaussian, which is by definition for everything but $q(x_{t-1}\|x_t, x_0)$ which is then Gaussian by the virtue of this equation. Plugging this back into the middle term in $L$ we get

$$ \mathbb{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} = \mathbb{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)} + \mathbb{E}_q \log\frac{q(x_{t-1}|x_0)}{q(x_{t}|x_0)}, $$

where from the second term, only the denominarot for $t=T$ and the numeratorfor $t=2$ survive the subsequent summation. This leads to

$$
\begin{align*}
L = &-\mathbb{E}_q\log \frac{p(x_T)}{q(x_T|x_0)} - \sum_{t=2}^T\mathbb{E}_q\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)} - \mathbb{E}_q\log p_\theta(x_0|x_1)=\\
&~ \mathbb{E}_qD_{KL}(q(x_T|x_0)\|p(x_T)) + \sum_{t=2}^T \mathbb{E}_qD_{KL}(q(x_{t-1}|x_t, x_0)\| p_{\theta}(x_{t-1}|x_t)) + \mathbb{E}_qCE(p_{\theta}(x_0|x_1)|| q(x_0, x_1) ) = \\
&~ L_T + \sum_{t=1}^{T-1} L_t + L_0
\end{align*}
$$

$L_T$ is the KL-divergence between two Gaussians and. Note that $q(x_T\|x_0)$ will not be perfectly an isotropic Gaussian other than the trivial case when $\beta_t = 1$ for some $t$ and just like with VAE-s, we sample the latent somewhat differently upon training and inference/sampling (but this term is trying to keep them close). If the $\beta_t$-s are not trainable then this term has no trainable component and can be dropped.

#### **The parametrization of the backwards model**

To optimize $L_t$, note that it is the KL-divergence between a Gaussian variable and $p_{\theta}(x_{t-1}\| x_t)$. This suggests choosing $p_{\theta}$ to be a Gaussian with parametrizable mean and variance:

$$ p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}, \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t)) \qquad 1<t\leq T.$$

This is particularly nice, because the KL-divergence between two Gaussians can be computed in closed form in terms of their parameters. All we need for that then is to compute said parameters of $q(x_{t-1}\|x_t, x_0)$:

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
\tilde\beta_t = \left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_t}\right)^{-1} = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$
$$
\tilde\mu(x_t, x_0) = \tilde\beta_t\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\bar{\alpha}_t}{1-\bar{\alpha_t}}x_0\right) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0.
$$
In the formula for the mean, we can eliminate $x_0$ by utilizing the representation $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar\alpha_t}\bar\varepsilon_t$ to get

$$\tilde\mu(x_t, x_0) = \tilde\mu(x_t, \bar\varepsilon_t) =  \frac{1}{\sqrt{\alpha_t}}\left(x_t -\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\bar\varepsilon_t\right)$$

#### **The final form of the objective**

This means that $q(x_{t-1}\|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde\mu(x_t, \bar\varepsilon_t), \tilde{\beta} I)$ and one can use the well known formula

$$ D_{KL}(\mathcal{N}_1, \mathcal{N}_2) = \frac{1}{2}\left[\mathrm{Tr}(\Sigma_2^{-1}\Sigma_1) - k + (\mu_2 -\mu_1)\Sigma_2^{-1}(\mu_2 -\mu_1) + \log\frac{\mathrm{det}\Sigma_1}{\mathrm{det}\Sigma_2}\right]$$

where $k$ is the dimension of $x_\cdot$ to arrive at

$$ L_t = \frac{1}{2}\left[\tilde{\beta}_t\mathrm{Tr}\Sigma^{-1}_\theta(x_t, t) -k + (\mu_{\theta}(x_t, t)-\tilde\mu(x_t, x_0))^T\Sigma_\theta^{-1}(x_t, t) (\mu_{\theta}(x_t, t)-\tilde\mu(x_t, x_0)) + k\log\tilde\beta_t - \log\det\Sigma_{\theta}(x_t, t)\right]. $$

Since the covariance matrix of the forward transition is diagonal, it is reasonable to postulate $\Sigma_\theta(x_t, t) = diag(\Sigma_{\theta}^{ii}(x_t, t))$, which simplifies this formula to a sum of one dimensional KL-divergences

$$ L_t = \frac{1}{2}\mathbb{E}_{x_0, x_t \sim q}\sum_i\left[ \frac{\tilde{\beta}_t + (\mu_{\theta}^i(x_t, t)-\tilde\mu^i(x_t, x_0))^2}{\Sigma_{\theta}^{ii}(x_t, t)} - \sum_i\log\Sigma_{\theta}^{ii}(x_t, t) - 1\right] + \frac{k(\log\tilde\beta_t -1)}{2}$$

This is where [Sohl-Dickstein et al (2015)](https://arxiv.org/pdf/1503.03585) effectively ends it. They set $(\mu_{\theta}(x_t, t), \sigma_{\theta}(x_t, t))$ to be a Neural Network (see below for architectures) and also learn $\beta_t$ by gradient descent. On the other hand, [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) set $\Sigma^{ii}_{\theta}(x_t, t)$ to be a constant $\sigma_t^2$ and treat the $\beta_t$-s (and thus the $\tilde \beta_t$-s) as hyperparameters. After dropping the now non-trainable terms, we end up with 

$$ L_t = \frac{1}{2\sigma_t^2}\mathbb{E}_{x_0, \bar\varepsilon_t}\|\tilde\mu(x_t, \bar\varepsilon_t) - \mu_{\theta}(x_t, t)\|^2 $$

that we could directly optimize, however, we can go even further using the definition of $\tilde\mu(x_t,\bar\varepsilon_t)$. To see this first rewrite the predicted mean as

$$
\mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha}_t}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_{\theta}(x_t, t)\right).
$$

Note that this is the definition of $\varepsilon_\theta$ so that it matches the formula we have for $\tilde\mu$. With this, $L_t$ above becomes
$$
L_t = \frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\sigma_t^2}\mathbb{E}_{x_0, \bar\varepsilon_t}\|\bar\varepsilon_t - \epsilon_{\theta}(x_t,t)\|^2 = \frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\sigma_t^2}\mathbb{E}_{x_0, \bar\varepsilon_t}\|\bar\varepsilon_t - \epsilon_{\theta}(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\bar\varepsilon_t,t)\|^2
$$

Other than the weighting factor before the expectation, this says that we want to train a network $\varepsilon_\theta$ that takes in a noisy image and an index of what stage of the denoising we are in and predicts the noise that was added to the original image. We mention that [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) reported better training result by dropping the time dependent pre-factor. This corresponds to the ELBO bound being replaced by a weighted version that emphasizes loss terms corresponding to larger $t$, the authors speculate that this helps because the amount of noise injected is smaller in the early steps so denoising these is an easier task that we do not need to enforce as strong.

#### **The decoder**

As we mentioned, [Ho et al (2020)](https://arxiv.org/pdf/2006.11239) derives a discrete decoder $p_{\theta}(x_0\|x_1)$ to obtain valid images assuming that all image data consists of integers in ${0, 1, \dots, 255}$ linearly scaled to $[-1, 1]$.

$$
p_{\theta}(x_0|x_1) = \prod_{i=1}^D\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}(x; \mu_{\theta}^i(x_1, 1), \sigma_1^2)dx
$$

$$
\delta_{\pm}(x) = \left\{\begin{array}{cc}
\pm\infty&\textrm{if}~x = \pm 1\\
x\pm\frac{1}{255}&\textrm{if}~\pm x < 1
\end{array} \right.
$$
The effect of this is simply to discretize the probability mass to multiples of $1/255$ corresponding to actual images.

#### **Summary**

This was a lot, let us recap. Diffusion models are latent variable models, where the latent variable is given by the trajectory of a backwards Markov Chain $z = x_{T:1}$. Unlike VAE-s, only the distribution of $x_T$ is specified by hand to be an isotropic Gaussian, the rest of the distribution come from a learnable Markov process. To learn this chain by the negative log-likelihood, we do importance sampling with the proposal distribution given by a very simple and tracktable Markov chain starting from $x_0$ and use the ELBO inequality to obtain a tracktable optimization objective. This bound then decomposes into boundary terms $L_T$, $L_0$ and internal terms $L_1,\dots, L_{T-1}$. The internal terms are all KL-divergences between the posterior of the forward transition and the backward transition. If the learnable model has Gaussian transitions, then we can explicitly write this KL-divergence down in terms of the means and variances (after assuming a diagonal covariance structure). If the covariances are isotropic, this will imply an $L^2$ loss on the predicted mean, what's more, suprisingly, we can take this even further when the learning problem becomes learning the noise in the noisy image under $L^2$ loss.

### (d) Training, Inference, Sampling

#### **Training**

Note that we share the neural network $\varepsilon_\theta, it is sufficient for each training step to sample a single timestep. The training loop thus consists of the following steps:

1. Choose a random element $x_0$ of the training set and sample an integer $t$ uniformly between $1$ and $T$.
2. Sample $\varepsilon\sim \mathcal{N}(0,I)$.
3. Take gradient descent step on $\nabla_\theta \|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon)\|$.

#### **Inference**

To evaluate the model probability of an unseen example, I believe we still have to do Monte-Carlo sampling to approximate

$$ p(x_0) = \mathbb{E}_{x_{1:T}\sim q(\cdot|x_0)}p(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}. $$

and is consequently fairly expensive consisting of the generation of $B$ trajectories and have $B*T$ neural network plus functions-on-top evaluations.

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








## Conditioned generation



## Architectures%   

## Resources

Besides the references in the text, I took inspiration from [Lilian Wang's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process) and the Stanford [Deep Generative Models course](https://deepgenerativemodels.github.io/).