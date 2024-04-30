# :construction: Retro Diffusion - a tour of diffusion models :construction:

This repo is an ongoing educational journey into diffusion models. The goal is to provide a
progressively expanding collection of PyTorch reference implementations of the most important
diffusion model milestones. I prefer to keep training and inference runnable locally on a laptop, I
will rely more on small popular datasets and occasional synthetic data. Then name implies that by
the time I finish, these models will be fairly "retro" (some of them already are). I plan to also write
some survey notes.

Papers (tentatively) covered:

| Name                                                               | Authors                   | ArXiv Link                                         | Year | Note                                                               |
|--------------------------------------------------------------------|---------------------------|----------------------------------------------------|------|--------------------------------------------------------------------|
| "Diffusion Probabilistic Models"                                   | Sohl-Dickstein et al      | [arXiv:1503.03585](https://arxiv.org/pdf/1503.03585) | 2015 | First paper that introduced the idea of diffusion models.         |
| "Denoising Diffusion Probabilistic Models"                         | Ho et al                  | [arXiv:2006.11239](https://arxiv.org/pdf/2006.11239) | 2020 | Added some crucial modifications of the original architecture.    |
| "Score-Based Generative Modeling through Stochastic Differential Equations" | Song et al.           | [arxiv:2011.13456](https://arxiv.org/pdf/2011.13456) | 2021 |                                                                    |
| "Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models" | Komanduri et. al | [arxiv:2404.17735](https://arxiv.org/pdf/2404.17735) | 2024 |

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
