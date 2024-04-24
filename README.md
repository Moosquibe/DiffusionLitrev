# Retro Diffusion - a tour through the history of diffusion models

This is an educational PyTorch Implementation of the first paper introducing diffusion models in *[Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)* by Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli published back in 2015. There is an implementation provided the authors at https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models but it is about 9 years old written using Theano so let's give it a facelift using PyTorch.

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

### Using make
<TODO>

