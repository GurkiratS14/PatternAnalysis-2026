"""
predict.py - Generate slices from a trained model and evaluate them.

Loads saved weights, generates an uncurated set of samples, times the
generation, and scores the samples with best-match SSIM against real slices.
Also saves the figures used for the memorisation audit.

Usage:
    python3 predict.py --model dcgan --weights ~/hipmri_runs/dcgan/dcgan.pt
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
from metrics import best_match_ssim
from modules import Generator

SSIM_TARGET = 0.6


def load_generator(weights_path, device):
    """Rebuild the DCGAN generator and load its trained weights."""
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


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate HipMRI slices.")
    parser.add_argument("--model", choices=["dcgan"], default="dcgan")
    parser.add_argument("--weights", default=None,
                        help="Defaults to ~/hipmri_runs/<model>/<model>.pt")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to ~/hipmri_runs/<model>/eval")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Use only the first N real slices per split (quick checks).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.expanduser(os.path.join("~/hipmri_runs", args.model))
    weights = args.weights or os.path.join(run_dir, f"{args.model}.pt")
    out_dir = args.out_dir or os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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

    results = {
        "model": args.model,
        "num_samples": len(samples),
        "ms_per_image": seconds_per_image * 1000,
        "generated_vs_test": summarise(gen_vs_test),
        "generated_vs_train": summarise(gen_vs_train),
        "real_test_vs_train": summarise(real_vs_train),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nBest-match SSIM (mean, std, min, max, share above "
          f"{SSIM_TARGET})")
    for name in ("generated_vs_test", "generated_vs_train", "real_test_vs_train"):
        s = results[name]
        print(f"  {name:20s} {s['mean']:.3f}  {s['std']:.3f}  {s['min']:.3f}  "
              f"{s['max']:.3f}  {s['fraction_above_target']:.0%}")

    plot_nearest_neighbours(samples, gen_vs_train, nearest, train_set,
                            os.path.join(out_dir, "nearest_neighbours.png"))
    plot_ssim_histogram(gen_vs_train, real_vs_train,
                        os.path.join(out_dir, "ssim_histogram.png"))
    print("Outputs saved to", out_dir)


if __name__ == "__main__":
    main()