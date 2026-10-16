"""
predict.py - Generate slices from a trained model and evaluate them.

For the GANs: generates an uncurated set of samples, times the generation,
scores them with best-match SSIM against real slices, counts near-copies,
measures diversity, and saves the memorisation-audit figures.

For the VQ-VAE: reconstructs every test slice, scores each against its own
original with paired SSIM, and saves the best and worst reconstructions.

For the PixelCNN prior: samples new code grids, decodes them with the VQ-VAE,
and runs the same sample audit as the GANs.

Usage:
    python3 predict.py --model dcgan
    python3 predict.py --model vqvae
    python3 predict.py --model pixelcnn
    python3 predict.py --model vqvae --interpolate   (also dcgan, wgan_gp)
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from dataset import DEFAULT_ROOT, HipMRISlices, check_no_patient_leakage
from metrics import best_match_ssim, mean_pairwise_ssim, ssim_pairs
from modules import VQVAE, Generator, PixelCNN

SSIM_TARGET = 0.6
SAMPLE_MODELS = ("dcgan", "wgan_gp", "pixelcnn")


def load_generator(weights_path, device):
    """Rebuild the GAN generator (shared by DCGAN and WGAN-GP) and load its weights."""
    checkpoint = torch.load(weights_path, map_location=device)
    generator = Generator(checkpoint["latent_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


@torch.no_grad()
def generate_samples(generator, num_samples, device, batch_size=64):
    """Generate images in [0, 1] and measure the time per image."""
    generator(torch.randn(batch_size, generator.latent_dim, device=device))  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()

    batches = []
    start = time.time()
    for first in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples - first)
        noise = torch.randn(count, generator.latent_dim, device=device)
        batches.append((generator(noise) + 1.0) / 2.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds_per_image = (time.time() - start) / num_samples
    return torch.cat(batches).cpu(), seconds_per_image


@torch.no_grad()
def generate_from_prior(weights_path, num_samples, device, temperature=1.0, batch_size=64):
    """Sample code grids from the PixelCNN prior and decode them with the VQ-VAE."""
    checkpoint = torch.load(weights_path, map_location=device)
    prior = PixelCNN(**checkpoint["config"]).to(device)
    prior.load_state_dict(checkpoint["model"])
    prior.eval()
    vq_checkpoint = torch.load(checkpoint["vqvae_weights"], map_location=device)
    vqvae = VQVAE(**vq_checkpoint["config"]).to(device)
    vqvae.load_state_dict(vq_checkpoint["model"])
    vqvae.eval()
    print(f"Loaded prior from epoch {checkpoint['epoch']} "
          f"({checkpoint['val_bits']:.3f} bits per code on validation)")

    height, width = checkpoint["grid"]
    batches = []
    start = time.time()
    for first in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples - first)
        codes = prior.sample(count, height, width, device, temperature)
        batches.append(vqvae.decode_indices(codes).clamp(0, 1).cpu())
        print(f"  sampled {first + count}/{num_samples}", flush=True)
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
            linewidth=2, color=blue, label="Unseen real slices (test patients)")
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
        axes[0, column].set_title(f"{label}: {test_set.patient_of(index)}\nSSIM {scores[index]:.2f}",
                                  fontsize=10)
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
    checkpoint = torch.load(weights, map_location=device)
    model = VQVAE(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
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
        recon, _, indices = model(batch)
        batches.append(recon.clamp(0, 1).cpu())
        codes.append(indices.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_image = (time.time() - start) / len(originals) * 1000

    recons = torch.cat(batches)
    scores = ssim_pairs(recons, originals)
    used = torch.cat(codes).unique().numel()
    results = {
        "model": "vqvae",
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
          f"{ms_per_image:.2f} ms per image, {used}/{results['codebook_size']} codes used")
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
    """Mid-level slices of each test patient, paired across patients, plus one within-patient pair."""
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
def interpolate_vqvae(vqvae, test_set, pairs, device, steps=8):
    """Blend two real slices' encoder outputs, re-quantise each blend, and decode it.

    Also returns, for each step, the share of grid positions whose code changed.
    """
    rows, changes = [], []
    weights = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)
    for first, second in pairs:
        latents = vqvae.encoder(torch.stack([test_set[first], test_set[second]]).to(device))
        blended = (1 - weights) * latents[0:1] + weights * latents[1:2]
        quantized, _, codes = vqvae.quantizer(blended)
        rows.append(vqvae.decoder(quantized).clamp(0, 1))
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

    if args.model == "vqvae":
        checkpoint = torch.load(weights, map_location=device)
        vqvae = VQVAE(**checkpoint["config"]).to(device)
        vqvae.load_state_dict(checkpoint["model"])
        vqvae.eval()
        test_set = HipMRISlices(args.root, "test", max_slices=args.max_slices)
        pairs = interpolation_pairs(test_set)
        rows, changes = interpolate_vqvae(vqvae, test_set, pairs, device)
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
          f"{results['crossfade_step_ssim_mean']:.3f}, min {results['crossfade_step_ssim_min']:.3f}")
    if "max_code_change_per_step" in results:
        print("Largest share of codes changing in one step, per path:",
              ", ".join(f"{c:.0%}" for c in results["max_code_change_per_step"]))
    print("Outputs saved to", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate HipMRI slices.")
    parser.add_argument("--model", choices=[*SAMPLE_MODELS, "vqvae"], default="dcgan")
    parser.add_argument("--weights", default=None,
                        help="Defaults to ~/hipmri_runs/<model>/<model>.pt")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to ~/hipmri_runs/<model>/eval")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Use only the first N real slices per split (quick checks).")
    parser.add_argument("--interpolate", action="store_true",
                        help="Save a latent interpolation figure instead of the usual evaluation.")
    parser.add_argument("--sample-batch", type=int, default=256,
                        help="PixelCNN: grids sampled in parallel (larger = faster per image).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="PixelCNN: sampling temperature (below 1 = safer, less varied).")
    parser.add_argument("--run-name", default=None,
                        help="Run folder under ~/hipmri_runs, default the model name.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.expanduser(os.path.join("~/hipmri_runs", args.run_name or args.model))
    weights = args.weights or os.path.join(run_dir, f"{args.model}.pt")
    eval_name = "eval" if args.temperature == 1.0 else f"eval_t{args.temperature:g}"
    out_dir = args.out_dir or os.path.join(run_dir, eval_name)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.interpolate:
        if args.model == "pixelcnn":
            raise ValueError("Interpolation uses the VQ-VAE latent space: run with --model vqvae.")
        run_interpolation(args, device, weights, run_dir)
        return

    if args.model == "vqvae":
        evaluate_vqvae(args, device, weights, out_dir)
        return

    if args.model == "pixelcnn":
        samples, seconds_per_image = generate_from_prior(weights, args.num_samples, device,
                                                         args.temperature, args.sample_batch)
    else:
        generator = load_generator(weights, device)
        samples, seconds_per_image = generate_samples(generator, args.num_samples, device)
    print(f"Generated {len(samples)} samples, "
          f"{seconds_per_image * 1000:.2f} ms per image")
    save_image(samples[:32], os.path.join(out_dir, "samples_uncurated.png"), nrow=4)

    check_no_patient_leakage(args.root)
    train_set = HipMRISlices(args.root, "train", max_slices=args.max_slices)
    test_set = HipMRISlices(args.root, "test", max_slices=args.max_slices)

    # Evenly spaced test slices: the reference for how close *unseen real*
    # patients sit to the training set.
    count = min(args.num_samples, len(test_set))
    picks = torch.linspace(0, len(test_set) - 1, count).long().tolist()
    real = torch.stack([test_set[i] for i in picks])

    gen_vs_test, _ = best_match_ssim(samples, test_set.images, device,
                                     description="Generated vs test")
    gen_vs_train, nearest = best_match_ssim(samples, train_set.images, device,
                                            description="Generated vs train")
    real_vs_train, _ = best_match_ssim(real, train_set.images, device,
                                       description="Real test vs train")

    # Samples closer to a training slice than any unseen real patient is.
    above_real_max = int((gen_vs_train > real_vs_train.max()).sum())

    results = {
        "model": args.model,
        "run": args.run_name or args.model,
        "temperature": args.temperature,
        "num_samples": len(samples),
        "ms_per_image": seconds_per_image * 1000,
        "generated_vs_test": summarise(gen_vs_test),
        "generated_vs_train": summarise(gen_vs_train),
        "real_test_vs_train": summarise(real_vs_train),
        "count_above_real_max": above_real_max,
        "pairwise_ssim_generated": mean_pairwise_ssim(samples, device),
        "pairwise_ssim_real_test": mean_pairwise_ssim(real, device),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nBest-match SSIM (mean, std, min, max, share above "
          f"{SSIM_TARGET})")
    for name in ("generated_vs_test", "generated_vs_train", "real_test_vs_train"):
        s = results[name]
        print(f"  {name:20s} {s['mean']:.3f}  {s['std']:.3f}  {s['min']:.3f}  "
              f"{s['max']:.3f}  {s['fraction_above_target']:.0%}")
    print(f"\nSamples closer to training than any real test slice: "
          f"{above_real_max}/{len(samples)}")
    print(f"Mean pairwise SSIM (diversity, lower = more varied): generated "
          f"{results['pairwise_ssim_generated']:.3f}, real test "
          f"{results['pairwise_ssim_real_test']:.3f}")

    plot_nearest_neighbours(samples, gen_vs_train, nearest, train_set,
                            os.path.join(out_dir, "nearest_neighbours.png"))
    plot_ssim_histogram(gen_vs_train, real_vs_train,
                        os.path.join(out_dir, "ssim_histogram.png"))
    print("Outputs saved to", out_dir)


if __name__ == "__main__":
    main()