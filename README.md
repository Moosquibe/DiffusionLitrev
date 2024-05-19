# :construction: Interactive Diffusion Literature Review - a tour of diffusion models :construction:

This repo is an ongoing educational journey into diffusion models. The goal is to provide a litreview with a progressively expanding collection of PyTorch reference implementations of the most important diffusion model milestones. Whenever possible, I prefer to keep training and inference runnable locally on an average modern MacBook, so I will rely more on small popular datasets and occasional synthetic data. Then name implies that by the time I finish, these models will be fairly "retro" (some of them already are). I am also writing a comprehensive [survey post](https://moosquibe.github.io/DiffusionLitrev/).

## Roadmap

I will first do a thorough literature review to have a better understanding of what is worth implementing. The result will be a deep dive survey [on a Github page](https://moosquibe.github.io/DiffusionLitrev/).

Then I will start making some educational implementations of the most important architectures. Here is a long list of papers I plan to cover tentatively. This list will probably be heavily edited or somewhat pruned. No promises.

| Name                                                               | Authors                   | ArXiv Link                                         | Year | Note                                                               |
|--------------------------------------------------------------------|---------------------------|----------------------------------------------------|------|--------------------------------------------------------------------|
| "Diffusion Probabilistic Models"                                   | Sohl-Dickstein et al      | [arXiv:1503.03585](https://arxiv.org/pdf/1503.03585) | 2015 | The paper that started it all, the first one to introduce the idea of diffusion models. |
| "Generative Modeling by Estimating Gradients of the Data Distribution" | Song & Ermon | [arxiv:1907.05600](https://arxiv.org/abs/1907.05600) | 2019 | Taking a somewhat parallel approach using Langevin Dynamics |
| "Denoising Diffusion Probabilistic Models" | Ho et al | [arXiv:2006.11239](https://arxiv.org/pdf/2006.11239) | 2020 | The one that made diffusions take off. Introduced predicting the noise instead of the reverse process mean, etc. |
| "Score-Based Generative Modeling through Stochastic Differential Equations" | Song et al. | [arxiv:2011.13456](https://arxiv.org/pdf/2011.13456) | 2021 | Experiments with continuous time SDE-s (Stochastic Differential Equations) for the generation. |
| "Diffusion Models Beat GANs on Image Synthesis" | Dhariwal & Nichol | [arxiv:2105.05233](https://arxiv.org/abs/2105.05233) | 2021 | Improve the architecture to achieve superior performance to GAN-s by the way of several ablations of the original setup. Also improves conditioned generation through classifier guidance. |
| "Classifier-Free Diffusion Guidance" | Ho & Salimans | [arxiv:2207.12598](https://arxiv.org/abs/2207.12598)| 2021 | Shows that diffusion guidance can be done without a classifier. |
| "Learning Transferable Visual Models From Natural Language Supervision" | Radford et al. | [arxiv:2103.00020](https://arxiv.org/abs/2103.00020) | 2021 | Introduces CLIP (Contrastive Language-Image Pre-training) which jointly trains representations between texts and images.
| "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models" | Nichol et. al | [arxiv:2112.10741](https://arxiv.org/abs/2112.10741) | 2022 | Compares CLIP and Classifier Free guidance |
| "High-Resolution Image Synthesis with Latent Diffusion Models" | Rombach et. al | [arxiv:2112.10752](https://arxiv.org/abs/2112.10752) | 2022 | Stable Diffusion: Latent Variable Diffusion Models |
| "Hierarchical Text-Conditional Image Generation with CLIP Latents" | Ho et al | [arxiv:2204.06125](https://arxiv.org/abs/2204.06125) | 2022 |CLIP + Latent Variables |
| "Denoising Diffusion Implicit Models" | Song et al. | [arxiv:2010.02502](https://arxiv.org/abs/2010.02502) | 2022 | Speeds up sampling through using a non-Markovian diffusion process. |
| "Progressive Distillation for Fast Sampling of Diffusion Models" | Salimans & Ho | [arxiv:2202.00512](https://arxiv.org/abs/2202.00512) | 2022 |Another approach to speed up sampling through distiallation (train a lighter student model on the output of a heavier teacher model) |
| "Consistency Models" | Song et al. | [arxiv:2303.01469](https://arxiv.org/abs/2303.01469) | 2023 | Yet another approach for speeding up sampling using models that directly map noise to data |
| "Scalable Diffusion Models with Transformers" | Peebles & Chie | [arxiv:2212.09748](https://arxiv.org/abs/2212.09748) | 2023 | Replace the U-Net in the network architecture with Transformers |
| "Adding Conditional Control to Text-to-Image Diffusion Models" | Zhang & al | [arxiv:2302.05543](https://arxiv.org/abs/2302.05543) | 2023 | Proposes ControlNet that adds strong conditioning control to Diffusion models. |
| "Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models" | Komanduri et. al | [arxiv:2404.17735](https://arxiv.org/pdf/2404.17735) | 2024 | Proposes a model for counterfactual generation according to a pre-specified causal model. |
| "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" | Lou et al. | [arxiv:2310.16834](https://arxiv.org/abs/2310.16834) | 2024 | Diffusion for LLM-s! |
| "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" | Esser et. al | [arxiv:2403.03206](https://arxiv.org/pdf/2403.03206) | 2024 | Stable Diffusion 3 |

## Usage

To read the main document, see

To run the notebooks

```bash
bazel run jupyterlab
```

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

To have access to requirements, add them to the jupyterlab target in `//tools/jupyter/BUILD`.
