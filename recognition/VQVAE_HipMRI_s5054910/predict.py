# --------------------------------------------------------------------
# File:    predict.py
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition, The University of Queensland
# Date:    October 2026
# Purpose: Evaluation driver: reconstruction SSIM, sampling with the
#          memorisation audit, and latent interpolation.
# --------------------------------------------------------------------
"""
predict.py - Generate slices from a trained model and evaluate them.

For a generative model (dcgan, wgan_gp, pixelcnn for VQ-VAE + prior, pixelcnn2
for VQ-VAE-2 + two priors): generates uncurated samples, times generation,
and runs the sample audit, optionally repeated over several seeds:
  * best-match SSIM of each sample against held-out real slices (fidelity) and
    against the training set (closeness to training patients);
  * the same for real held-out slices, as the reference for an unseen patient;
  * near-copies: samples closer to a training slice than any held-out slice;
  * diversity: mean pairwise SSIM at full and at 4x-reduced resolution.
--reference picks the real held-out patients the audit compares against:
validate (4 patients) for choosing models and temperatures, test (3 patients)
for reporting final numbers, or heldout (both, 7 patients). Keeping these
separate means test patients never influence a modelling decision.

For an autoencoder (vqvae, vqvae2): reconstructs every test slice, scores
each against its own original with paired SSIM, and saves the best and worst.

--interpolate saves a latent interpolation figure (dcgan, wgan_gp, vqvae, vqvae2).

Usage:
    python3 predict.py --model dcgan --repeats 3
    python3 predict.py --model vqvae
    python3 predict.py --model pixelcnn --run-name pixelcnn_reg --temperature 0.9 --repeats 3
    python3 predict.py --model pixelcnn2 --repeats 3 --reference validate --temperature 0.95
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
from metrics import best_match_ssim, mean_pairwise_ssim, ssim_pairs
from modules import VQVAE, VQVAE2, Generator, PixelCNN

SSIM_TARGET = 0.6
SAMPLE_MODELS = ("dcgan", "wgan_gp", "pixelcnn", "pixelcnn2")
AUTOENCODERS = ("vqvae", "vqvae2")
# Real held-out splits used as the audit reference for each --reference choice.
REFERENCE_SPLITS = {"validate": ("validate",), "test": ("test",),
                    "heldout": ("test", "validate")}


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
    kind = checkpoint.get("kind", "vqvae")
    model = (VQVAE2 if kind == "vqvae2" else VQVAE)(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def load_prior(weights_path, device):
    """Rebuild a trained PixelCNN prior in evaluation mode."""
    checkpoint = torch.load(weights_path, map_location=device)
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

    prior, checkpoint = load_prior(weights, device)
    vqvae, _ = load_autoencoder(checkpoint["vqvae_weights"], device)
    height, width = checkpoint["grid"]

    if args.model == "pixelcnn":
        def sample_prior(count):
            """Sample `count` code grids from the prior and decode them."""
            codes = prior.sample(count, height, width, device, args.temperature)
            return vqvae.decode_levels([codes]).clamp(0, 1)
        return sample_prior

    # VQ-VAE-2: sample a top-level layout, then bottom-level detail conditioned on it.
    top_weights = args.top_weights or os.path.expanduser(
        "~/hipmri_runs/pixelcnn2_top/pixelcnn.pt")
    top_prior, top_checkpoint = load_prior(top_weights, device)
    top_height, top_width = top_checkpoint["grid"]

    def sample_two_level(count):
        """Sample a top layout, then bottom detail conditioned on it, and decode."""
        top = top_prior.sample(count, top_height, top_width, device, args.temperature)
        bottom = prior.sample(count, height, width, device, args.temperature, cond_codes=top)
        return vqvae.decode_levels([top, bottom]).clamp(0, 1)
    return sample_two_level


@torch.no_grad()
def generate(sampler, num_samples, batch_size, device):
    """Generate num_samples images in batches and measure the time per image."""
    batches = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for first in range(0, num_samples, batch_size):
        batches.append(sampler(min(batch_size, num_samples - first)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(batches), (time.time() - start) / num_samples


def summarise(scores):
    """Summary statistics for a set of best-match SSIM scores."""
    return {
        "mean": scores.mean().item(),
        "std": scores.std().item(),
        "min": scores.min().item(),
        "max": scores.max().item(),
        "fraction_above_target": (scores > SSIM_TARGET).float().mean().item(),
    }


def plot_nearest_neighbours(samples, scores, indices, train_set, path, count=6):
    """Show the generated images that are closest to a training slice."""
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
        axes[2, column].imshow(error, cmap="magma", vmin=0, vmax=0.5)
    for ax in axes.flat:
        ax.axis("off")
    for row, name in enumerate(["Original", "Reconstruction", "Absolute error"]):
        axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_vqvae(args, device, weights, out_dir):
    """Reconstruct every test slice and score it against its own original."""
    model, checkpoint = load_autoencoder(weights, device)
    print(f"Loaded epoch {checkpoint['epoch']} (validation SSIM {checkpoint['val_ssim']:.3f})")

    check_no_patient_leakage(args.root)
    test_set = HipMRISlices(args.root, "test", max_slices=args.max_slices)
    originals = torch.stack([test_set[i] for i in range(len(test_set))])

    model(originals[:8].to(device))  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    batches, codes = [], []
    start = time.time()
    for first in range(0, len(originals), 64):
        batch = originals[first:first + 64].to(device)
        batches.append(model(batch)[0].clamp(0, 1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_image = (time.time() - start) / len(originals) * 1000

    recons = torch.cat(batches)
    scores = ssim_pairs(recons, originals)
    for first in range(0, len(originals), 64):
        levels = model.encode_levels(originals[first:first + 64].to(device))
        codes.append([level.cpu() for level in levels])
    used = [torch.cat([batch[i] for batch in codes]).unique().numel() for i in range(len(codes[0]))]
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
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

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


def slerp(start, end, t):
    """Spherical interpolation between two noise vectors (keeps them at a typical length)."""
    omega = torch.acos((start / start.norm() * end / end.norm()).sum().clamp(-1, 1))
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
        model, _ = load_autoencoder(weights, device)
        test_set = HipMRISlices(args.root, "test", max_slices=args.max_slices)
        pairs = interpolation_pairs(test_set)
        rows, changes = interpolate_autoencoder(model, test_set, pairs, device)
        results["pairs"] = [[test_set.patient_of(a), test_set.patient_of(b)] for a, b in pairs]
        results["max_code_change_per_step"] = changes.max(dim=1).values.tolist()
    else:
        rows = interpolate_gan(load_generator(weights, device), device)

    fades = crossfade(rows)
    model_steps, fade_steps = step_similarity(rows), step_similarity(fades)
    results.update({
        "model_step_ssim_mean": model_steps.mean().item(),
        "model_step_ssim_min": model_steps.min().item(),
        "crossfade_step_ssim_mean": fade_steps.mean().item(),
        "crossfade_step_ssim_min": fade_steps.min().item(),
        "model_step_ssim_per_path": model_steps.tolist(),
    })
    with open(os.path.join(out_dir, "interpolation.json"), "w") as f:
        json.dump(results, f, indent=2)

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


def coarse(images, factor=4):
    """4x average-pooled images: diversity of overall layout, insensitive to fine artefacts."""
    return F.avg_pool2d(images, factor)


def mean_and_std(values):
    """Mean and sample standard deviation of per-seed results."""
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"mean": tensor.mean().item(),
            "std": tensor.std().item() if len(values) > 1 else 0.0,
            "values": values}


def run_sample_audit(args, device, weights, out_dir):
    """Generate samples (over several seeds if --repeats > 1) and audit each set."""
    sampler = make_sampler(args, weights, device)
    batch_size = args.sample_batch if args.model.startswith("pixelcnn") else 64

    check_no_patient_leakage(args.root)
    train_set = HipMRISlices(args.root, "train", max_slices=args.max_slices)
    # Selection uses validation patients only; reporting uses test patients only.
    heldout_sets = [HipMRISlices(args.root, split, max_slices=args.max_slices)
                    for split in REFERENCE_SPLITS[args.reference]]
    heldout = torch.cat([torch.stack([s[i] for i in range(len(s))]) for s in heldout_sets])
    # Evenly spaced held-out slices: the reference for an unseen real patient.
    picks = torch.linspace(0, len(heldout) - 1, min(args.num_samples, len(heldout))).long()
    real = heldout[picks]
    real_vs_train, _ = best_match_ssim(real, train_set.images, device,
                                       description="Real held-out vs train")
    real_max = real_vs_train.max()

    runs, seconds_per_image = [], 0.0
    for repeat in range(args.repeats):
        torch.manual_seed(args.seed + repeat)
        samples, seconds = generate(sampler, args.num_samples, batch_size, device)
        if repeat == 0:
            seconds_per_image = seconds
            save_image(samples[:32], os.path.join(out_dir, "samples_uncurated.png"), nrow=4)
        run, gen_vs_train, nearest = audit_samples(samples, train_set, heldout, device, real_max)
        runs.append(run)
        print(f"Seed {args.seed + repeat}: near-copies {run['count_above_real_max']}/"
              f"{len(samples)}, vs train {run['generated_vs_train']['mean']:.3f}, "
              f"vs held-out {run['generated_vs_heldout']['mean']:.3f}, "
              f"pairwise {run['pairwise_ssim']:.3f} (coarse {run['pairwise_ssim_coarse']:.3f})",
              flush=True)
        if repeat == 0:
            plot_nearest_neighbours(samples, gen_vs_train, nearest, train_set,
                                    os.path.join(out_dir, "nearest_neighbours.png"))
            plot_ssim_histogram(gen_vs_train, real_vs_train,
                                os.path.join(out_dir, "ssim_histogram.png"))

    results = {
        "model": args.model,
        "run": args.run_name or args.model,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "reference": args.reference,
        "seeds": [args.seed + r for r in range(args.repeats)],
        "ms_per_image": seconds_per_image * 1000,
        "heldout_patients": len({s.patient_of(i) for s in heldout_sets for i in range(len(s))}),
        "real_heldout_vs_train": summarise(real_vs_train),
        "pairwise_ssim_real": mean_pairwise_ssim(real, device),
        "pairwise_ssim_coarse_real": mean_pairwise_ssim(coarse(real), device),
        "repeats": {
            "count_above_real_max": mean_and_std([r["count_above_real_max"] for r in runs]),
            "generated_vs_train_mean":
                mean_and_std([r["generated_vs_train"]["mean"] for r in runs]),
            "generated_vs_train_max": mean_and_std([r["generated_vs_train"]["max"] for r in runs]),
            "generated_vs_heldout_mean": mean_and_std(
                [r["generated_vs_heldout"]["mean"] for r in runs]),
            "pairwise_ssim": mean_and_std([r["pairwise_ssim"] for r in runs]),
            "pairwise_ssim_coarse": mean_and_std([r["pairwise_ssim_coarse"] for r in runs]),
        },
        "per_seed": runs,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    r, s = results["repeats"], results["real_heldout_vs_train"]
    print(f"\nGenerated {args.num_samples} samples per seed, "
          f"{seconds_per_image * 1000:.2f} ms per image")
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


def main():
    """Parse command-line arguments and run the requested evaluation."""
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
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat sampling and the audit with this many seeds.")
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Use only the first N real slices per split (quick checks).")
    parser.add_argument("--interpolate", action="store_true",
                        help="Save a latent interpolation figure instead of the usual evaluation.")
    parser.add_argument("--sample-batch", type=int, default=256,
                        help="PixelCNN: grids sampled in parallel.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="PixelCNN: sampling temperature (below 1 = safer, less varied).")
    parser.add_argument("--run-name", default=None,
                        help="Run folder under ~/hipmri_runs, default the model name.")
    parser.add_argument("--reference", choices=sorted(REFERENCE_SPLITS), default="test",
                        help="Audit reference: validate to choose, test to report, "
                             "heldout for both pooled.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.expanduser(os.path.join("~/hipmri_runs", args.run_name or args.model))
    filename = "pixelcnn.pt" if args.model.startswith("pixelcnn") else f"{args.model}.pt"
    weights = args.weights or os.path.join(run_dir, filename)
    eval_name = "eval" if args.temperature == 1.0 else f"eval_t{args.temperature:g}"
    if args.model in SAMPLE_MODELS and args.reference != "heldout":
        eval_name += f"_{args.reference}"  # keep earlier pooled-reference results intact
    out_dir = args.out_dir or os.path.join(run_dir, eval_name)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.interpolate:
        if args.model.startswith("pixelcnn"):
            raise ValueError("Interpolation uses the autoencoder's latent space: "
                             "run with --model vqvae or vqvae2.")
        run_interpolation(args, device, weights, run_dir)
    elif args.model in AUTOENCODERS:
        evaluate_vqvae(args, device, weights, out_dir)
    else:
        run_sample_audit(args, device, weights, out_dir)


if __name__ == "__main__":
    main()
