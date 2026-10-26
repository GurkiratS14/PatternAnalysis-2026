# --------------------------------------------------------------------
# File:    predict.py
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Evaluation driver: reconstruction SSIM, FID / KID generation
#          quality, the memorisation audit, and latent interpolation.
# --------------------------------------------------------------------
"""
predict.py - Generate slices from a trained model and evaluate them.

Four evaluation modes:

* Reconstruction (--model vqvae / vqvae2): reconstructs every test slice and
  scores it against its own original with paired SSIM (the >= 0.6 target).
* Generation quality (--fid, generative models): FID and KID between generated
  slices and the real slices of the chosen reference patients, with the same
  scores for random real training slices as the attainable floor. Samples are
  cached on disk, so a later run on another reference reuses them. With
  --model vqvae / vqvae2, --fid scores the reconstructions of those training
  slices instead: the best FID the decoder allows, before any prior is involved.
* Memorisation audit (default for generative models): how close each sample is
  to its most similar training slice compared with real unseen patients
  (near-copies), and diversity as mean pairwise SSIM. SSIM is used here only to
  detect copying, never as a quality target.
* Interpolation (--interpolate): a latent interpolation figure and its
  frame-to-frame smoothness.

--reference picks the real held-out patients used as the reference:
validate (4 patients) for choosing models and temperatures, test (3 patients)
for reporting final numbers, or heldout (both, 7 patients). Keeping these
separate means test patients never influence a modelling decision.

Generative models: dcgan, wgan_gp, pixelcnn (VQ-VAE + prior) and pixelcnn2
(VQ-VAE-2 + two priors).

Usage:
    python3 predict.py --model vqvae2
    python3 predict.py --model pixelcnn2 --fid --repeats 3 --temperature 0.95 --reference validate
    python3 predict.py --model pixelcnn2 --repeats 3 --temperature 0.95 --reference validate
    python3 predict.py --model dcgan --fid --repeats 3 --reference test
    python3 predict.py --model vqvae2 --fid --repeats 3 --reference validate
    python3 predict.py --model vqvae2 --interpolate
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from dataset import DEFAULT_ROOT, HipMRISlices, check_no_patient_leakage
from metrics import (best_match_ssim, frechet_distance, inception_features,
                     kernel_inception_distance, mean_pairwise_ssim, ssim_pairs)
from modules import VQVAE, VQVAE2, Generator, PixelCNN

SSIM_TARGET = 0.6
SAMPLE_MODELS = ("dcgan", "wgan_gp", "pixelcnn", "pixelcnn2")
AUTOENCODERS = ("vqvae", "vqvae2")
# Real held-out splits used as the reference for each --reference choice.
REFERENCE_SPLITS = {"validate": ("validate",), "test": ("test",),
                    "heldout": ("test", "validate")}


# ----------------------------------------------------------------------------
# Loading models and sampling
# ----------------------------------------------------------------------------

def load_generator(weights_path, device):
    """Rebuild the GAN generator (shared by DCGAN and WGAN-GP) and load its weights."""
    checkpoint = torch.load(weights_path, map_location=device)
    generator = Generator(checkpoint["latent_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


def load_autoencoder(weights_path, device):
    """Rebuild a trained VQ-VAE or VQ-VAE-2 in evaluation mode."""
    checkpoint = torch.load(weights_path, map_location=device)
    # Checkpoints saved before VQ-VAE-2 existed have no "kind" and are single-level.
    kind = checkpoint.get("kind", "vqvae")
    model = (VQVAE2 if kind == "vqvae2" else VQVAE)(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def load_prior(weights_path, device):
    """Rebuild a trained PixelCNN prior in evaluation mode."""
    checkpoint = torch.load(os.path.expanduser(weights_path), map_location=device)
    prior = PixelCNN(**checkpoint["config"]).to(device)
    prior.load_state_dict(checkpoint["model"])
    print(f"Loaded {checkpoint.get('level', 'single')} prior from epoch {checkpoint['epoch']} "
          f"({checkpoint['val_bits']:.3f} bits per code on validation)")
    return prior.eval(), checkpoint


def make_sampler(args, weights, device):
    """Return a function that generates `count` images in [0, 1] from the chosen model."""
    if args.model in ("dcgan", "wgan_gp"):
        generator = load_generator(weights, device)

        def sample_gan(count):
            """Draw `count` generator samples, rescaled from [-1, 1] to [0, 1]."""
            noise = torch.randn(count, generator.latent_dim, device=device)
            return (generator(noise) + 1.0) / 2.0
        return sample_gan

    # A prior checkpoint records which autoencoder its codes came from.
    prior, checkpoint = load_prior(weights, device)
    vqvae, _ = load_autoencoder(checkpoint["vqvae_weights"], device)
    height, width = checkpoint["grid"]

    if args.model == "pixelcnn":
        def sample_prior(count):
            """Sample `count` code grids from the prior and decode them."""
            codes = prior.sample(count, height, width, device, args.temperature,
                                 top_k=args.top_k, top_p=args.top_p)
            return vqvae.decode_levels([codes]).clamp(0, 1)
        return sample_prior

    # VQ-VAE-2: sample a top-level layout, then bottom-level detail conditioned on it.
    top_weights = args.top_weights or os.path.expanduser(
        "~/hipmri_runs/pixelcnn2_top/pixelcnn.pt")
    top_prior, top_checkpoint = load_prior(top_weights, device)
    top_height, top_width = top_checkpoint["grid"]

    def sample_two_level(count):
        """Sample a top layout, then bottom detail conditioned on it, and decode."""
        # The top level decides the anatomy, so it can be given its own temperature.
        top_temperature = args.top_temperature or args.temperature
        top = top_prior.sample(count, top_height, top_width, device, top_temperature,
                               top_k=args.top_k, top_p=args.top_p)
        bottom = prior.sample(count, height, width, device, args.temperature, cond_codes=top,
                              top_k=args.top_k, top_p=args.top_p)
        return vqvae.decode_levels([top, bottom]).clamp(0, 1)
    return sample_two_level


@torch.no_grad()
def generate(sampler, num_samples, batch_size, device):
    """Generate num_samples images in batches and measure the time per image."""
    batches = []
    # GPU work is asynchronous, so synchronise before reading the clock.
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for first in range(0, num_samples, batch_size):
        batches.append(sampler(min(batch_size, num_samples - first)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(batches), (time.time() - start) / num_samples


def sample_batch_size(args):
    """PixelCNN grids are sampled many at once; GANs are fast at any batch size."""
    return args.sample_batch if args.model.startswith("pixelcnn") else 64


def load_split(args, split):
    """All slices of one split as a single (N, 1, 128, 256) tensor, plus the dataset."""
    dataset = HipMRISlices(args.root, split, max_slices=args.max_slices)
    return torch.stack([dataset[i] for i in range(len(dataset))]), dataset


def load_reference(args):
    """Real slices of the --reference patients, and the datasets they came from."""
    # Selection uses validation patients only; reporting uses test patients only.
    loaded = [load_split(args, split) for split in REFERENCE_SPLITS[args.reference]]
    images = torch.cat([images for images, _ in loaded])
    return images, [dataset for _, dataset in loaded]


def count_patients(datasets):
    """Number of distinct patients across one or more datasets."""
    return len({d.patient_of(i) for d in datasets for i in range(len(d))})


# ----------------------------------------------------------------------------
# Summary helpers and figures
# ----------------------------------------------------------------------------

def summarise(scores):
    """Summary statistics for a set of SSIM scores."""
    return {
        "mean": scores.mean().item(),
        "std": scores.std().item(),
        "min": scores.min().item(),
        "max": scores.max().item(),
        "fraction_above_target": (scores > SSIM_TARGET).float().mean().item(),
    }


def mean_and_std(values):
    """Mean and sample standard deviation of per-seed results."""
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"mean": tensor.mean().item(),
            "std": tensor.std().item() if len(values) > 1 else 0.0,
            "values": values}


def save_json(results, path):
    """Write a results dictionary as indented JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def plot_nearest_neighbours(samples, scores, indices, train_set, path, count=6):
    """Show the generated images that are closest to a training slice."""
    # The most similar samples are the ones most likely to be copies, so show those.
    order = scores.argsort(descending=True)[:count]
    fig, axes = plt.subplots(2, count, figsize=(3 * count, 4.4), squeeze=False)
    for column, sample_index in enumerate(order):
        train_index = indices[sample_index].item()
        axes[0, column].imshow(samples[sample_index, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"Generated (SSIM {scores[sample_index]:.2f})", fontsize=10)
        axes[1, column].imshow(train_set[train_index][0], cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title(f"Nearest: {train_set.patient_of(train_index)}", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Generated slices most similar to a training slice")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ssim_histogram(generated_scores, real_scores, path):
    """Compare how close generated and unseen real slices sit to the training set."""
    blue, orange = "#2a78d6", "#eb6834"
    # A shared range keeps the two histograms on the same bins.
    low = min(generated_scores.min(), real_scores.min()).item()
    high = max(generated_scores.max(), real_scores.max()).item()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(real_scores.numpy(), bins=30, range=(low, high), histtype="step",
            linewidth=2, color=blue, label="Held-out real slices (unseen patients)")
    ax.hist(generated_scores.numpy(), bins=30, range=(low, high), histtype="step",
            linewidth=2, color=orange, linestyle="--", label="Generated slices")
    ax.set(title="Similarity to the most similar training slice",
           xlabel="Best-match SSIM against the training set",
           ylabel="Number of slices")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstructions(originals, recons, scores, test_set, path, count=3):
    """Original, reconstruction and absolute error for the best and worst slices."""
    order = scores.argsort()
    chosen = torch.cat([order[-count:].flip(0), order[:count]])
    fig, axes = plt.subplots(3, len(chosen), figsize=(3 * len(chosen), 5.4), squeeze=False)
    for column, index in enumerate(chosen.tolist()):
        label = "Best" if column < count else "Worst"
        error = (originals[index, 0] - recons[index, 0]).abs()
        axes[0, column].imshow(originals[index, 0], cmap="gray", vmin=0, vmax=1)
        title = f"{label}: {test_set.patient_of(index)}\nSSIM {scores[index]:.2f}"
        axes[0, column].set_title(title, fontsize=10)
        axes[1, column].imshow(recons[index, 0], cmap="gray", vmin=0, vmax=1)
        # Errors above 0.5 are rare, so a 0-0.5 colour scale shows small errors clearly.
        axes[2, column].imshow(error, cmap="magma", vmin=0, vmax=0.5)
    for ax in axes.flat:
        ax.axis("off")
    for row, name in enumerate(["Original", "Reconstruction", "Absolute error"]):
        axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Reconstruction (the SSIM >= 0.6 target)
# ----------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_all(model, originals, device, batch_size=64):
    """Reconstruct every slice and measure the time per image."""
    model(originals[:8].to(device))  # warm-up, so one-off CUDA set-up is not timed
    if device.type == "cuda":
        torch.cuda.synchronize()
    batches = []
    start = time.time()
    for first in range(0, len(originals), batch_size):
        batch = originals[first:first + batch_size].to(device)
        batches.append(model(batch)[0].clamp(0, 1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(batches), (time.time() - start) / len(originals) * 1000


@torch.no_grad()
def codes_used_per_level(model, originals, device, batch_size=64):
    """Distinct codebook entries the test slices use, at each level."""
    codes = []
    for first in range(0, len(originals), batch_size):
        levels = model.encode_levels(originals[first:first + batch_size].to(device))
        codes.append([level.cpu() for level in levels])
    return [torch.cat([batch[i] for batch in codes]).unique().numel()
            for i in range(len(codes[0]))]


@torch.no_grad()
def evaluate_vqvae(args, device, weights, out_dir):
    """Reconstruct every test slice and score it against its own original."""
    model, checkpoint = load_autoencoder(weights, device)
    print(f"Loaded epoch {checkpoint['epoch']} (validation SSIM {checkpoint['val_ssim']:.3f})")

    check_no_patient_leakage(args.root)
    originals, test_set = load_split(args, "test")
    recons, ms_per_image = reconstruct_all(model, originals, device)
    # Paired SSIM: each test slice against its own reconstruction.
    scores = ssim_pairs(recons, originals)
    used = codes_used_per_level(model, originals, device)

    results = {
        "model": args.model,
        "checkpoint_epoch": checkpoint["epoch"],
        "num_test_slices": len(originals),
        "ms_per_image": ms_per_image,
        "reconstruction_ssim": summarise(scores),
        "reconstruction_mse": torch.mean((recons - originals) ** 2).item(),
        "codes_used_on_test": used,
        "codebook_size": checkpoint["config"]["num_embeddings"],
    }
    save_json(results, os.path.join(out_dir, "metrics.json"))

    s = results["reconstruction_ssim"]
    print(f"\nReconstruction SSIM on {len(originals)} test slices: mean {s['mean']:.3f}, "
          f"std {s['std']:.3f}, min {s['min']:.3f}, max {s['max']:.3f}, "
          f"{s['fraction_above_target']:.0%} above {SSIM_TARGET}")
    print(f"Reconstruction MSE {results['reconstruction_mse']:.5f}, "
          f"{ms_per_image:.2f} ms per image, codes used per level {used} "
          f"of {results['codebook_size']}")
    plot_reconstructions(originals, recons, scores, test_set,
                         os.path.join(out_dir, "reconstructions.png"))
    print("Outputs saved to", out_dir)


# ----------------------------------------------------------------------------
# Latent interpolation
# ----------------------------------------------------------------------------

def slerp(start, end, t):
    """Spherical interpolation between two noise vectors (keeps them at a typical length)."""
    omega = torch.acos((start / start.norm() * end / end.norm()).sum().clamp(-1, 1))
    # Nearly parallel vectors: slerp reduces to ordinary linear interpolation.
    if omega.abs() < 1e-6:
        return (1 - t) * start + t * end
    return (torch.sin((1 - t) * omega) * start + torch.sin(t * omega) * end) / torch.sin(omega)


def interpolation_pairs(test_set):
    """Mid-level test slices paired across patients, plus one within-patient pair."""
    by_patient = {}
    for index in range(len(test_set)):
        by_patient.setdefault(test_set.patient_of(index), []).append(index)
    groups = list(by_patient.values())
    mids = [group[len(group) // 2] for group in groups]
    pairs = list(zip(mids, mids[1:])) + [(mids[0], mids[-1])]
    pairs.append((groups[0][len(groups[0]) // 4], groups[0][3 * len(groups[0]) // 4]))
    return list(dict.fromkeys(pairs))  # drop duplicates, keep order


@torch.no_grad()
def interpolate_gan(generator, device, num_pairs=4, steps=8):
    """Generate images along slerp paths between random pairs of noise vectors."""
    rows = []
    for _ in range(num_pairs):
        start, end = torch.randn(2, generator.latent_dim, device=device)
        noise = torch.stack([slerp(start, end, t) for t in torch.linspace(0, 1, steps)])
        rows.append((generator(noise) + 1.0) / 2.0)
    return torch.stack(rows).cpu()


@torch.no_grad()
def interpolate_autoencoder(model, test_set, pairs, device, steps=8):
    """Blend two real slices in the model's latent space, re-quantise and decode each blend.

    Also returns, for each step, the share of (bottom) grid positions whose code changed.
    """
    rows, changes = [], []
    weights = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)
    for first, second in pairs:
        pair = torch.stack([test_set[first], test_set[second]]).to(device)
        images, codes = model.interpolate(pair, weights)
        rows.append(images.clamp(0, 1))
        # A large share of codes changing in one step means a jump, not a smooth path.
        changes.append((codes[1:] != codes[:-1]).float().mean(dim=(1, 2)))
    return torch.stack(rows).cpu(), torch.stack(changes).cpu()


def step_similarity(rows):
    """SSIM between consecutive frames of each interpolation path: (paths, steps - 1)."""
    return torch.stack([ssim_pairs(row[:-1], row[1:]) for row in rows])


def crossfade(rows):
    """Pixel-space fade between each path's first and last frame, as a reference."""
    steps = rows.size(1)
    weights = torch.linspace(0, 1, steps).view(1, -1, 1, 1, 1)
    return (1 - weights) * rows[:, :1] + weights * rows[:, -1:]


def run_interpolation(args, device, weights, run_dir):
    """Latent interpolation figure and smoothness statistics for one model."""
    out_dir = os.path.join(run_dir, "interpolation")
    os.makedirs(out_dir, exist_ok=True)
    results: dict = {"model": args.model}

    if args.model in AUTOENCODERS:
        # Autoencoders interpolate between real test slices; GANs between random noise.
        model, _ = load_autoencoder(weights, device)
        test_set = HipMRISlices(args.root, "test", max_slices=args.max_slices)
        pairs = interpolation_pairs(test_set)
        rows, changes = interpolate_autoencoder(model, test_set, pairs, device)
        results["pairs"] = [[test_set.patient_of(a), test_set.patient_of(b)] for a, b in pairs]
        results["max_code_change_per_step"] = changes.max(dim=1).values.tolist()
    else:
        rows = interpolate_gan(load_generator(weights, device), device)

    # A pixel cross-fade of the same end frames is the "no learned structure" baseline.
    fades = crossfade(rows)
    model_steps, fade_steps = step_similarity(rows), step_similarity(fades)
    results.update({
        "model_step_ssim_mean": model_steps.mean().item(),
        "model_step_ssim_min": model_steps.min().item(),
        "crossfade_step_ssim_mean": fade_steps.mean().item(),
        "crossfade_step_ssim_min": fade_steps.min().item(),
        "model_step_ssim_per_path": model_steps.tolist(),
    })
    save_json(results, os.path.join(out_dir, "interpolation.json"))

    # Each model path, followed by the pixel cross-fade between the same two end frames.
    grid = torch.stack([rows, fades], dim=1).flatten(0, 2)
    save_image(grid, os.path.join(out_dir, "interpolation.png"), nrow=rows.size(1), padding=4,
               pad_value=1.0)
    print(f"Consecutive-frame SSIM: model mean {results['model_step_ssim_mean']:.3f}, "
          f"min {results['model_step_ssim_min']:.3f}; cross-fade mean "
          f"{results['crossfade_step_ssim_mean']:.3f}, "
          f"min {results['crossfade_step_ssim_min']:.3f}")
    if "max_code_change_per_step" in results:
        print("Largest share of codes changing in one step, per path:",
              ", ".join(f"{c:.0%}" for c in results["max_code_change_per_step"]))
    print("Outputs saved to", out_dir)


# ----------------------------------------------------------------------------
# Memorisation and diversity audit (SSIM used only to detect copying)
# ----------------------------------------------------------------------------

def coarse(images, factor=4):
    """4x average-pooled images: diversity of overall layout, insensitive to fine artefacts."""
    return F.avg_pool2d(images, factor)


def audit_samples(samples, train_set, heldout, device, real_max):
    """Best-match SSIM, near-copy count and diversity for one set of generated samples."""
    gen_vs_heldout, _ = best_match_ssim(samples, heldout, device,
                                        description="Generated vs held-out")
    gen_vs_train, nearest = best_match_ssim(samples, train_set.images, device,
                                            description="Generated vs train")
    return {
        "generated_vs_heldout": summarise(gen_vs_heldout),
        "generated_vs_train": summarise(gen_vs_train),
        # Samples closer to a training slice than any held-out real patient is.
        "count_above_real_max": int((gen_vs_train > real_max).sum()),
        "pairwise_ssim": mean_pairwise_ssim(samples, device),
        "pairwise_ssim_coarse": mean_pairwise_ssim(coarse(samples), device),
    }, gen_vs_train, nearest


def real_baseline(heldout, train_set, num_samples, device):
    """How close real unseen slices sit to the training set: the reference for near-copies."""
    # Evenly spaced held-out slices, as many as the generated samples per seed.
    picks = torch.linspace(0, len(heldout) - 1, min(num_samples, len(heldout))).long()
    real = heldout[picks]
    real_vs_train, _ = best_match_ssim(real, train_set.images, device,
                                       description="Real held-out vs train")
    return real, real_vs_train


def summarise_audit(args, runs, heldout_sets, real, real_vs_train, seconds, device):
    """Collect the per-seed audits into one results dictionary (mean +/- std over seeds)."""
    def over_seeds(key, field=None):
        """Mean and std over seeds of one audit value (a nested field if given)."""
        return mean_and_std([r[key][field] if field else r[key] for r in runs])

    return {
        "model": args.model,
        "run": args.run_name or args.model,
        "temperature": args.temperature,
        "sampling": sampling_tag(args) or "full",
        "num_samples": args.num_samples,
        "reference": args.reference,
        "seeds": [args.seed + r for r in range(args.repeats)],
        "ms_per_image": seconds * 1000,
        "heldout_patients": count_patients(heldout_sets),
        "real_heldout_vs_train": summarise(real_vs_train),
        "pairwise_ssim_real": mean_pairwise_ssim(real, device),
        "pairwise_ssim_coarse_real": mean_pairwise_ssim(coarse(real), device),
        "repeats": {
            "count_above_real_max": over_seeds("count_above_real_max"),
            "generated_vs_train_mean": over_seeds("generated_vs_train", "mean"),
            "generated_vs_train_max": over_seeds("generated_vs_train", "max"),
            "generated_vs_heldout_mean": over_seeds("generated_vs_heldout", "mean"),
            "pairwise_ssim": over_seeds("pairwise_ssim"),
            "pairwise_ssim_coarse": over_seeds("pairwise_ssim_coarse"),
        },
        "per_seed": runs,
    }


def print_audit(results, args, out_dir):
    """Print the audit summary in the format used for the README tables."""
    r, s = results["repeats"], results["real_heldout_vs_train"]
    print(f"\nGenerated {args.num_samples} samples per seed, "
          f"{results['ms_per_image']:.2f} ms per image")
    print(f"Reference ({args.reference}): {results['heldout_patients']} held-out patients, "
          f"best-match SSIM vs train "
          f"mean {s['mean']:.3f}, max {s['max']:.3f}")
    print(f"Over {args.repeats} seed(s), mean +/- std:")
    for key in ("generated_vs_heldout_mean", "generated_vs_train_mean", "generated_vs_train_max",
                "count_above_real_max", "pairwise_ssim", "pairwise_ssim_coarse"):
        print(f"  {key:28s} {r[key]['mean']:.3f} +/- {r[key]['std']:.3f}")
    print(f"  {'pairwise_ssim (real)':28s} {results['pairwise_ssim_real']:.3f}")
    print(f"  {'pairwise_ssim_coarse (real)':28s} {results['pairwise_ssim_coarse_real']:.3f}")
    print("Outputs saved to", out_dir)


def run_sample_audit(args, device, weights, out_dir):
    """Generate samples (over several seeds if --repeats > 1) and audit each set."""
    sampler = make_sampler(args, weights, device)
    check_no_patient_leakage(args.root)
    train_set = HipMRISlices(args.root, "train", max_slices=args.max_slices)
    heldout, heldout_sets = load_reference(args)
    real, real_vs_train = real_baseline(heldout, train_set, args.num_samples, device)
    real_max = real_vs_train.max()

    runs, seconds_per_image = [], 0.0
    for repeat in range(args.repeats):
        # One seed per repeat, so the spread over seeds shows sampling noise.
        torch.manual_seed(args.seed + repeat)
        samples, seconds = generate(sampler, args.num_samples, sample_batch_size(args), device)
        run, gen_vs_train, nearest = audit_samples(samples, train_set, heldout, device, real_max)
        runs.append(run)
        print(f"Seed {args.seed + repeat}: near-copies {run['count_above_real_max']}/"
              f"{len(samples)}, vs train {run['generated_vs_train']['mean']:.3f}, "
              f"vs held-out {run['generated_vs_heldout']['mean']:.3f}, "
              f"pairwise {run['pairwise_ssim']:.3f} (coarse {run['pairwise_ssim_coarse']:.3f})",
              flush=True)
        # Figures and timing come from the first seed only.
        if repeat == 0:
            seconds_per_image = seconds
            save_image(samples[:32], os.path.join(out_dir, "samples_uncurated.png"), nrow=4)
            plot_nearest_neighbours(samples, gen_vs_train, nearest, train_set,
                                    os.path.join(out_dir, "nearest_neighbours.png"))
            plot_ssim_histogram(gen_vs_train, real_vs_train,
                                os.path.join(out_dir, "ssim_histogram.png"))

    results = summarise_audit(args, runs, heldout_sets, real, real_vs_train,
                              seconds_per_image, device)
    save_json(results, os.path.join(out_dir, "metrics.json"))
    print_audit(results, args, out_dir)


# ----------------------------------------------------------------------------
# Generation quality: FID and KID
# ----------------------------------------------------------------------------

def cached_samples(args, device, weights, run_dir):
    """Generated samples for every seed, loaded from disk if they were made before.

    Samples do not depend on the reference, so the expensive PixelCNN sampling
    runs once and both the validation and the test scoring reuse it.
    """
    cache_dir = os.path.join(run_dir, "samples")
    os.makedirs(cache_dir, exist_ok=True)
    name = (f"samples_t{args.temperature:g}{sampling_tag(args)}_seed{args.seed}_"
            f"{args.repeats}x{args.num_samples}.pt")
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        print("Loaded cached samples from", path)
        return torch.load(path).float() / 255.0, None

    sampler = make_sampler(args, weights, device)
    batches, seconds = [], 0.0
    for repeat in range(args.repeats):
        # Same seeds as the audit, so FID and the audit describe the same samples.
        torch.manual_seed(args.seed + repeat)
        samples, seconds_per_image = generate(sampler, args.num_samples,
                                              sample_batch_size(args), device)
        batches.append(samples)
        seconds += seconds_per_image / args.repeats
    samples = torch.cat(batches)
    # Stored as 8-bit images: 4x smaller on disk, and the Inception network only
    # ever sees 8-bit images in the standard FID protocol anyway.
    torch.save((samples * 255).round().to(torch.uint8), path)
    print("Saved samples to", path)
    return (samples * 255).round() / 255.0, seconds


def train_floor_subset(train_images, count, seed):
    """Random real training slices (fixed seed): the floor that FID is compared with."""
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randperm(len(train_images), generator=generator)[:count]
    return train_images[picks]


def score_against_reference(named_sets, reference, device):
    """FID and KID of each named image set against the reference slices."""
    # 8-bit quantisation for every set, so all of them are treated identically.
    sets = {name: (images * 255).round() / 255.0 for name, images in named_sets.items()}
    reference_features = inception_features((reference * 255).round() / 255.0, device)
    scores = {}
    for name, images in sets.items():
        features = inception_features(images, device)
        kid_mean, kid_std = kernel_inception_distance(features, reference_features)
        scores[name] = {"fid": frechet_distance(features, reference_features),
                        "kid": kid_mean, "kid_std": kid_std}
    return scores


def print_scores(results, labels, out_dir):
    """FID and KID lines in the format used for the README tables (KID x 1000)."""
    print(f"\nReference ({results['reference']}): {results['reference_patients']} patients, "
          f"{results['num_reference']} slices; {results['num_scored']} scored slices per set")
    for name, label in labels:
        r = results[name]
        print(f"  {label:24s} FID {r['fid']:7.2f}   "
              f"KID x1000 {1000 * r['kid']:6.2f} +/- {1000 * r['kid_std']:.2f}")
    print("Outputs saved to", out_dir)


def run_fid(args, device, weights, run_dir, out_dir):
    """FID and KID of generated samples against the --reference patients, with a real floor."""
    samples, seconds = cached_samples(args, device, weights, run_dir)
    check_no_patient_leakage(args.root)
    reference, reference_sets = load_reference(args)
    train_images, _ = load_split(args, "train")

    # Floor: random real training slices, as many as the samples. Even real data from
    # other patients does not reach FID 0, so this is the best a generator can expect.
    train_subset = train_floor_subset(train_images, len(samples), args.seed)
    scores = score_against_reference({"generated": samples, "train": train_subset},
                                     reference, device)

    results = {"model": args.model, "run": args.run_name or args.model,
               "temperature": args.temperature, "sampling": sampling_tag(args) or "full",
               "reference": args.reference,
               "reference_patients": count_patients(reference_sets),
               "num_scored": len(samples), "num_reference": len(reference),
               "seeds": [args.seed + r for r in range(args.repeats)], **scores}
    if seconds is not None:
        results["ms_per_image"] = seconds * 1000
    save_json(results, os.path.join(out_dir, "fid.json"))
    print_scores(results, (("generated", "Generated"), ("train", "Real train (floor)")),
                 out_dir)


def run_reconstruction_fid(args, device, weights, out_dir):
    """FID and KID of the autoencoder's reconstructions: the best its decoder allows.

    The same random training slices as the floor in run_fid are scored twice, as
    they are and after encoding and decoding. The gap between the two is the cost
    of the autoencoder alone; whatever remains between the reconstructions and the
    prior's samples is the cost of the prior.
    """
    model, _ = load_autoencoder(weights, device)
    check_no_patient_leakage(args.root)
    reference, reference_sets = load_reference(args)
    train_images, _ = load_split(args, "train")
    train_subset = train_floor_subset(train_images, args.repeats * args.num_samples, args.seed)
    recons, _ = reconstruct_all(model, train_subset, device)
    scores = score_against_reference({"reconstructed": recons, "train": train_subset},
                                     reference, device)

    results = {"model": args.model, "reference": args.reference,
               "reference_patients": count_patients(reference_sets),
               "num_scored": len(train_subset), "num_reference": len(reference), **scores}
    save_json(results, os.path.join(out_dir, "reconstruction_fid.json"))
    print_scores(results, (("reconstructed", "Reconstructed train"),
                           ("train", "Real train (floor)")), out_dir)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def parse_args():
    """Command-line options for every evaluation mode."""
    parser = argparse.ArgumentParser(description="Generate and evaluate HipMRI slices.")
    parser.add_argument("--model", choices=[*SAMPLE_MODELS, *AUTOENCODERS], default="dcgan")
    parser.add_argument("--weights", default=None,
                        help="Defaults to ~/hipmri_runs/<run>/<model>.pt (pixelcnn.pt for priors)")
    parser.add_argument("--top-weights", default=None,
                        help="pixelcnn2: top-level prior, "
                             "default ~/hipmri_runs/pixelcnn2_top/pixelcnn.pt")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to ~/hipmri_runs/<run>/eval")
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Generated samples per seed.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat sampling with this many seeds.")
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Use only the first N real slices per split (quick checks).")
    parser.add_argument("--interpolate", action="store_true",
                        help="Save a latent interpolation figure instead of the usual evaluation.")
    parser.add_argument("--fid", action="store_true",
                        help="FID and KID instead of the audit (for vqvae / vqvae2: of the "
                             "reconstructions, the best the decoder allows).")
    parser.add_argument("--sample-batch", type=int, default=256,
                        help="PixelCNN: grids sampled in parallel.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="PixelCNN: sampling temperature (below 1 = safer, less varied).")
    parser.add_argument("--top-k", type=int, default=0,
                        help="PixelCNN: sample only from the k most likely codes (0 = all).")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="PixelCNN: nucleus sampling, keep codes up to this total "
                             "probability (1.0 = all).")
    parser.add_argument("--top-temperature", type=float, default=None,
                        help="VQ-VAE-2: separate temperature for the top (layout) prior, "
                             "default the same as --temperature.")
    parser.add_argument("--run-name", default=None,
                        help="Run folder under ~/hipmri_runs, default the model name.")
    parser.add_argument("--reference", choices=sorted(REFERENCE_SPLITS), default="test",
                        help="Real patients to compare against: validate to choose, test to "
                             "report, heldout for both pooled.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sampling_tag(args):
    """Suffix naming the non-default sampling options, e.g. '_k32_tt0.8' ('' for defaults)."""
    tag = ""
    if args.top_k > 0:
        tag += f"_k{args.top_k}"
    if args.top_p < 1.0:
        tag += f"_p{args.top_p:g}"
    if args.top_temperature is not None:
        tag += f"_tt{args.top_temperature:g}"
    return tag


def output_dir(args, run_dir):
    """eval[_t<T>][_<reference>] inside the run folder, unless --out-dir is given."""
    if args.out_dir:
        return args.out_dir
    name = "eval" if args.temperature == 1.0 else f"eval_t{args.temperature:g}"
    name += sampling_tag(args)
    if (args.model in SAMPLE_MODELS or args.fid) and args.reference != "heldout":
        name += f"_{args.reference}"  # keep earlier pooled-reference results intact
    return os.path.join(run_dir, name)


def main():
    """Parse command-line arguments and run the requested evaluation."""
    args = parse_args()
    run_dir = os.path.expanduser(os.path.join("~/hipmri_runs", args.run_name or args.model))
    filename = "pixelcnn.pt" if args.model.startswith("pixelcnn") else f"{args.model}.pt"
    weights = args.weights or os.path.join(run_dir, filename)
    out_dir = output_dir(args, run_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Fixed seed so a rerun draws the same noise and the same samples.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.interpolate:
        if args.model.startswith("pixelcnn"):
            raise ValueError("Interpolation uses the autoencoder's latent space: "
                             "run with --model vqvae or vqvae2.")
        run_interpolation(args, device, weights, run_dir)
    elif args.model in AUTOENCODERS and args.fid:
        run_reconstruction_fid(args, device, weights, out_dir)
    elif args.model in AUTOENCODERS:
        evaluate_vqvae(args, device, weights, out_dir)
    elif args.fid:
        run_fid(args, device, weights, run_dir, out_dir)
    else:
        run_sample_audit(args, device, weights, out_dir)


if __name__ == "__main__":
    main()
