# --------------------------------------------------------------------
# File:    sample_figures.py
# Author:  Gurkirat Singh Randhawa (50549102)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: README figures and checks from the cached evaluation samples:
#          uncurated grids across all seeds, and a high-frequency artefact
#          measurement of generated versus real test slices.
# --------------------------------------------------------------------
"""
Works only from files predict.py has already written, so it needs no GPU and
never samples again. predict.py --fid stores every seed's samples in
~/hipmri_runs/<run>/samples/ as one uint8 tensor (seed 0, then 1, then 2).

    python3 sample_figures.py            # writes into ./figures/

Outputs:
    figures/<name>_seeds.png    first 8 samples of each seed, unselected
    figures/zoom_strip.png      same 64 x 64 corner of real and generated slices
    figures/high_frequency.json share of spectral power above 0.25 cycles/pixel
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import DEFAULT_ROOT, HipMRISlices

RUNS = os.path.expanduser("~/hipmri_runs")
# Display name -> cached samples file (3 seeds x 256 samples each).
MODELS = {
    "dcgan_aug": "dcgan_aug/samples/samples_t1_seed0_3x256.pt",
    "wgan_gp": "wgan_gp/samples/samples_t1_seed0_3x256.pt",
    "vqvae2_aug_t0.9": "pixelcnn2_aug/samples/samples_t0.9_seed0_3x256.pt",
    "vqvae2_aug_k32": "pixelcnn2_aug/samples/samples_t1_k32_seed0_3x256.pt",
}
SEEDS, PER_SEED, ZOOM = 3, 8, (slice(8, 72), slice(0, 64))


def load_samples(path):
    """Cached uint8 samples as a float (N, 1, 128, 256) tensor in [0, 1]."""
    return torch.load(path).float() / 255.0


def seed_grid(samples, title, path):
    """First PER_SEED samples of every seed, two rows of four per seed."""
    per_seed = len(samples) // SEEDS
    fig, axes = plt.subplots(SEEDS * 2, 4, figsize=(8, 6.3))
    for seed in range(SEEDS):
        for k in range(PER_SEED):
            ax = axes[seed * 2 + k // 4, k % 4]
            ax.imshow(samples[seed * per_seed + k, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if k % 4 == 0:
                ax.set_ylabel(f"seed {seed}" if k == 0 else "", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.94, bottom=0.005, wspace=0.03, hspace=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def high_frequency_share(images, cutoff=0.25):
    """Fraction of each image's spectral power above `cutoff` cycles per pixel.

    Images are quantised to 8 bits first, as the cached samples are, so both
    sides carry the same quantisation noise. Returns (mean, std) over images.
    """
    x = (images[:, 0] * 255).round() / 255
    x = x - x.mean(dim=(1, 2), keepdim=True)
    power = torch.fft.fft2(x).abs() ** 2
    fy = torch.fft.fftfreq(x.shape[1]).abs()[:, None]
    fx = torch.fft.fftfreq(x.shape[2]).abs()[None, :]
    high = (fy ** 2 + fx ** 2).sqrt() > cutoff
    share = (power * high).sum(dim=(1, 2)) / power.sum(dim=(1, 2))
    return share.mean().item(), share.std().item()


def zoom_strip(rows, path, columns=6):
    """The same 64 x 64 corner (left fat band) of a few slices per row, at native pixels."""
    fig, axes = plt.subplots(len(rows), columns, figsize=(columns * 1.6, len(rows) * 1.75))
    generator = torch.Generator().manual_seed(0)
    for r, (name, images) in enumerate(rows.items()):
        picks = torch.randperm(len(images), generator=generator)[:columns]
        for c, i in enumerate(picks):
            ax = axes[r, c]
            ax.imshow(images[i, 0][ZOOM], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.99, bottom=0.01, wspace=0.04, hspace=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Make the seed grids, the zoom strip and the high-frequency table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    test = HipMRISlices(args.root, "test")
    real = torch.stack([test[i] for i in range(len(test))])
    results = {"real_test": high_frequency_share(real)}
    strip = {"real test": real}

    for name, relative in MODELS.items():
        path = os.path.join(RUNS, relative)
        if not os.path.exists(path):
            print("missing, skipped:", path)
            continue
        samples = load_samples(path)
        results[name] = high_frequency_share(samples)
        seed_grid(samples, f"{name}: first {PER_SEED} samples of each seed (uncurated)",
                  os.path.join(args.out_dir, f"{name}_seeds.png"))
        if name in ("dcgan_aug", "vqvae2_aug_t0.9"):
            strip[name] = samples

    zoom_strip(strip, os.path.join(args.out_dir, "zoom_strip.png"))
    with open(os.path.join(args.out_dir, "high_frequency.json"), "w") as f:
        json.dump(results, f, indent=2)
    for name, (mean, std) in results.items():
        print(f"{name:18s} high-frequency power share {mean:.5f} +/- {std:.5f}")


if __name__ == "__main__":
    main()
