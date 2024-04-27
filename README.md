# :construction: Retro Diffusion - a tour of diffusion models :construction:

This repo is an ongoing educational journey into diffusion models. The goal is to provide a
progressively expanding collection of PyTorch reference implementations of the most important
diffusion model milestones. I prefer to keep training and inference runnable locally on a laptop, I
will rely more on small popular datasets and occasional synthetic data. Then name implies that by
the time I finish, these models will be fairly "retro" (some of them already are).

I start with the first paper introducing diffusion models in
_[Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)_
by Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli published back in 2015. There is an
implementation provided the authors at
https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models but it is about 9 years old written
using Theano so let's give it a facelift using PyTorch.

## Install and setup

### Using Bazel

**To update package versions:**

Start by updating the version `requirements.in` and then run

```bash
bazel run requirements.update()
```

The result can be validated by

```bash
bazel test requirements_test
```
