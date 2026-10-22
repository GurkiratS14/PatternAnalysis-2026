---
title: Synthetic Prostate MRI Slices with VQ-VAE-2 and PixelCNN Priors (HipMRI)
author: Gurkirat Singh Randhawa
student_id: s5054910
course: COMP3710 Pattern Recognition, The University of Queensland
task: 2.4 Image Generation (Hard)
date: 2026-10-29
---

# Synthetic Prostate MRI Slices with VQ-VAE-2 and PixelCNN Priors (HipMRI)

**Gurkirat Singh Randhawa · s5054910 · COMP3710 Pattern Recognition · Task 2.4 Image Generation · Hard difficulty**

## Contents
1. [Problem and algorithm](#1-problem-and-algorithm)
2. [How it works](#2-how-it-works)
3. [Feasibility review](#3-feasibility-review)
4. [Dependencies and reproducibility](#4-dependencies-and-reproducibility)
5. [Usage, file structure, example inputs and outputs](#5-usage-file-structure-example-inputs-and-outputs)
6. [Pre-processing and data splits](#6-pre-processing-and-data-splits)
7. [Open Research Dilemma](#7-open-research-dilemma)
8. [Discussion and conclusion](#8-discussion-and-conclusion)
9. [Artificial Intelligence Usage Disclosure](#9-artificial-intelligence-usage-disclosure)
10. [References](#10-references)

## 1. Problem and algorithm

**The problem.** A research group holding the HipMRI prostate study wants a **synthetic cohort**: realistic 2D pelvic MRI slices that it can share between centres or use for augmentation, **without exposing real patients**. A useful generator has to meet three goals at once:
- **Realism:** slices look like real MRI.
- **Diversity:** no mode collapse.
- **Privacy:** no copying of training patients.

**The algorithm.** A **VQ-VAE-2** (Razavi et al., 2019) compresses each 128×256 slice into two grids of discrete codes:
- a coarse **top grid** (16×32) for the overall anatomy;
- a fine **bottom grid** (32×64) for texture.

Two **gated PixelCNN priors** (van den Oord et al., 2016) learn which code grids are plausible. A new slice is made by sampling a top grid, then a bottom grid conditioned on it, and decoding both.

**Baselines.** The model is benchmarked on identical patient-level splits against a single-level **VQ-VAE** (van den Oord et al., 2017) and two GANs, **DCGAN** and **WGAN-GP**. The two strongest families, DCGAN and VQ-VAE-2, are also trained with left-right flip augmentation as an ablation.

**Everything is implemented from scratch in PyTorch:** all models, the PixelCNN and the SSIM, FID and KID code. No pre-trained weights are used anywhere except the standard Inception-v3 inside FID, which is used only to *measure* images, as staff permitted on Ed.

**Headline results** (test patients, scored once after all choices were made on validation):

| | VQ-VAE-2 + aug (Hard model) | DCGAN + aug (best baseline) |
|---|---|---|
| Reconstruction SSIM (target > 0.6) | **0.919**, 100 % of slices > 0.6 | — (no encoder) |
| FID / KID×1000 (real floor 50.9 / 10.1) | 127.0 / 112.5 | **86.1 / 51.0** |
| Near-copies of training slices | **0.4 %** | **0.4 %** |
| Diversity, pairwise SSIM (real 0.193) | 0.176 | 0.161 |
| Sampling time per slice | 1.48 s | 0.18 ms |

![Uncurated VQ-VAE-2 + aug samples](images/vqvae2_aug_samples_t0.9.png)
*Figure 1. 32 uncurated VQ-VAE-2 + aug samples at temperature 0.9 (test run, seed 0).*

## 2. How it works

![Pipeline](images/pipeline.png)
*Figure 2. Stage 1 trains the autoencoder, stage 2 trains the priors on its codes, and stage 3 generates new slices.*

### 2.1 Vector quantisation

The encoder turns an image into a grid of 64-dimensional vectors z_e. Each vector is replaced by its nearest entry in a codebook of 512 learned vectors:

  k = argmin_j ‖z_e − e_j‖²,  z_q = e_k

*Worked example.* Take z_e = (0.9, 0.1) and a codebook e₀ = (1, 0), e₁ = (0, 1). The squared distances are 0.02 and 1.62, so the vector becomes code 0.

**Loss** (sg = stop-gradient, β = 0.25):

  L = ‖x − x̂‖² + ‖sg[z_e] − e‖² + β‖z_e − sg[e]‖²

| Term | Effect |
|---|---|
| ‖x − x̂‖² | Reconstruction: the output should match the input |
| ‖sg[z_e] − e‖² | Moves the codes towards the encoder outputs |
| β‖z_e − sg[e]‖² | Keeps the encoder outputs close to their codes |

- **Straight-through estimator.** argmin has no gradient, so the decoder's gradient is copied to the encoder: `z_q = z_e + (z_q − z_e).detach()`.
- **Dead-code restarts** (Dhariwal et al., 2020). Codes that no input selects are moved onto recent encoder outputs. This raised usage from 3–4 of 512 codes to 507–512.

### 2.2 Why two levels

In a single-level VQ-VAE each code covers a 4×4-pixel patch, so the prior places 2,048 codes one by one with no plan for the global layout. In practice its samples had realistic texture but jumbled anatomy (§7.6, case 1).

VQ-VAE-2 adds a **top level whose codes each cover 8×8 pixels**. The top prior decides where the bladder, hips and rectum go first. The bottom level is quantised *with the decoded top level as context*, so its codes only need to add detail. This one change improves FID by about 45 at every temperature (§7.2).

### 2.3 Gated PixelCNN prior

The prior factorises the distribution of a code grid c in raster order (left to right, top to bottom):

  p(c) = ∏ᵢ p(cᵢ | c₁ … cᵢ₋₁)

Each factor is a 512-way softmax.

- **Network:** 12 gated layers with 128 channels, 7×7 kernels and gate tanh(a)·σ(b).
- **Two stacks:** a **vertical stack** sees the rows above and a **horizontal stack** sees the positions to the left. Together they remove the original PixelCNN's blind spot.
- **First-layer mask:** hides the current position, so no prediction sees its own answer. A test proves this (`tests/test_pixelcnn_causality.py`).
- **Conditioning:** the bottom prior embeds the top codes, upsamples them 2× and adds them before every gate.

**Temperature T.** Logits are divided by T before the softmax. *Example:* logits [2, 1, 0] give probabilities [0.67, 0.24, 0.09] at T = 1 and [0.87, 0.12, 0.02] at T = 0.5. Lower T gives safer, less varied samples.

### 2.4 Baselines

| | DCGAN (Radford et al., 2015) | WGAN-GP (Gulrajani et al., 2017) |
|---|---|---|
| Generator (shared) | z ∈ ℝ¹⁰⁰ → 4×8 block → 5 transposed convs → tanh | same |
| Discriminator / critic | 5 strided convs with batch norm | same layout with **layer norm** (the penalty is computed per sample) |
| Loss | non-saturating BCE | Wasserstein loss + gradient penalty, λ = 10, n_critic = 5 |
| Optimiser | Adam 2e-4, β = (0.5, 0.999), 100 epochs | Adam 1e-4, β = (0, 0.9), 500 epochs |

### 2.5 Hyperparameters and their justification

| Setting | Value | Why |
|---|---|---|
| Codebook | 512 codes × 64 dims per level | Large enough that 507–512 codes are used (perplexity ≈ 330), with no collapse |
| Commitment β | 0.25 | Value from the VQ-VAE paper; reconstruction was stable |
| Autoencoder optimiser | Adam 2e-4, batch 64, 100 epochs, best validation SSIM kept | Validation SSIM was still rising slowly at epoch 100 |
| Prior | 12 layers, 128 channels, 7×7 kernels | Receptive field covers most of a 32×64 grid within GPU memory |
| Prior regularisation | dropout 0.3, AdamW weight decay 0.01 | The prior overfits 31 patients within a few epochs (§7.4) |
| Prior batch | 32 | Keeps peak memory under the 8 GB budget (4.7 GB) |
| Early stopping | patience 10 on validation loss/SSIM | Stops once validation stops improving; the best epoch is kept |
| Gradient clipping | norm 1.0 (priors) | Stops one unusual batch from destabilising PixelCNN training |
| Augmentation | random left-right flip (training only) | Pelvic MRI is roughly symmetric; it doubles the data for a small cohort |
| Sampling | 3 seeds × 256 samples | Gives mean ± std, so single lucky runs don't decide anything |

## 3. Feasibility review

*One-page review, also available as [`feasibility_review.pdf`](feasibility_review.pdf).*

> **Problem and goal.** Generate realistic 2D prostate MRI slices that a research group could share or use for augmentation without exposing real patients. Check whether a VQ-VAE-2 with learned priors gives slices that are realistic, diverse and not copied from training, compared with GAN baselines.
>
> **Acceptance criteria.**
> 1. Reconstruction: test SSIM > 0.6.
> 2. Quality: FID / KID close to the real-data floor.
> 3. No memorisation: ≤ 5 % of samples closer to a training slice than any real unseen patient.
> 4. Diversity: no mode collapse (pairwise SSIM ≈ real).
> 5. Comparison: against DCGAN and WGAN-GP on the same patient split.
> 6. Resources: ≤ 2 h training per model, ≤ 8 GB GPU memory.
>
> **Approach.**
> - **Model:** VQ-VAE-2 compresses each slice into two grids of discrete codes, a coarse layout grid and a fine detail grid. Two PixelCNN priors learn which code grids are plausible; new slices come from sampling the priors and decoding.
> - **Baselines:** DCGAN and WGAN-GP.
> - **Data:** HipMRI, 38 patients, split 31 / 4 / 3 by patient (no leakage). Settings are chosen on validation; final numbers come from test.
>
> **Evidence so far.**
> - Reconstruction SSIM 0.90 on test, with every slice above 0.6.
> - The two-level model clearly improves anatomy over a single-level VQ-VAE (about 45 FID better).
> - About 1 % near-copies for VQ-VAE-2, against 9 % for WGAN-GP.
> - All models train in under 2 h at under 5 GB GPU memory.
>
> **Risks and fallback.**
>
> | Risk | Plan |
> |---|---|
> | Prior overfits (small dataset) | Dropout, weight decay, augmentation, early stopping |
> | Sampling is slow (about 1.5 s per slice) | Accept as a trade-off; faster prior as future work |
> | GAN baseline scores better FID | Report honestly and compare trade-offs |
>
> **Fallback:** WGAN-GP with the same evaluation (Normal difficulty).

The final outcome against each criterion is in §7.7.

## 4. Dependencies and reproducibility

| Package | Version | Used for |
|---|---|---|
| Python | 3.11.15 | |
| PyTorch | 2.13.0 (+cu130) | models and training |
| torchvision | 0.28.0 | image grids |
| numpy | 2.4.6 | arrays |
| nibabel | 5.4.2 | reading NIfTI slices |
| matplotlib | 3.11.2 | figures |
| tqdm | 4.70.1 | progress bars |
| pillow | 12.3.0 | image saving |
| pytorch-fid | 0.3.0 | Inception-v3 weights for FID/KID only |
| scikit-image, scipy | 0.26.0, 1.17.1 | tests that check SSIM and FID against reference implementations |
| pytest | 9.1.1 | running `tests/` |

Install PyTorch for your CUDA version first, then:
```bash
pip install -r requirements.txt
pip install --no-deps pytorch-fid==0.3.0   # --no-deps keeps the CUDA build of PyTorch
```
Hardware: one NVIDIA A100 on UQ's Rangpur cluster (`comp3710` partition). The Slurm scripts set `TORCH_HOME` so the compute nodes find the FID weights.

**Reproducibility.**
- **Fixed seeds:** 42 for training; 0, 1 and 2 for sampling.
- **Fixed splits:** the provided patient folders, with a leakage check every time data is loaded.
- **Tested:** retraining the VQ-VAE-2 top prior with the same seed reproduced it **bit for bit**. The bottom prior matched to within ±0.005 bits per code, the remaining difference coming from non-deterministic GPU kernels.
- **Reported spread:** every generation metric is a mean ± std over 3 seeds, because near-copy counts alone varied by 0–3 between runs.
- **Logs:** the training and evaluation logs behind every number are in [`logs/`](logs/).
- **Weights:** not committed (size); the commands in §5 recreate the Hard model in about 1 A100-hour.
