"""
train.py - Training script for HipMRI image generation models.

Currently trains the DCGAN baseline. Saves sample grids, loss curves and
model weights to --out-dir, which should sit outside the repository.

Usage:
    python3 train.py --model dcgan --epochs 100
Quick check on a few slices:
    python3 train.py --model dcgan --epochs 1 --max-slices 64 --batch-size 16
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import save_image

from dataset import DEFAULT_ROOT, get_dataloaders
from modules import LATENT_DIM, Discriminator, Generator, count_parameters


def to_tanh_range(images):
    """[0, 1] -> [-1, 1], to match the generator's tanh output."""
    return images * 2.0 - 1.0


def to_unit_range(images):
    """[-1, 1] -> [0, 1], for saving and for metrics."""
    return (images + 1.0) / 2.0


def save_history(history, out_dir):
    """Write per-epoch training statistics to a CSV file and a PNG figure."""
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    blue, orange = "#2a78d6", "#eb6834"
    fig, (ax_loss, ax_prob) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [row["d_loss"] for row in history], color=blue,
                 linewidth=2, label="Discriminator")
    ax_loss.plot(epochs, [row["g_loss"] for row in history], color=orange,
                 linewidth=2, linestyle="--", label="Generator")
    ax_loss.set(title="Training loss", xlabel="Epoch", ylabel="BCE loss")

    ax_prob.plot(epochs, [row["d_real"] for row in history], color=blue,
                 linewidth=2, label="D(x): real images")
    ax_prob.plot(epochs, [row["d_fake"] for row in history], color=orange,
                 linewidth=2, linestyle="--", label="D(G(z)): fake images")
    ax_prob.axhline(0.5, color="gray", linewidth=1, linestyle=":")
    ax_prob.set(title="Discriminator's belief that an image is real",
                xlabel="Epoch", ylabel="Mean probability", ylim=(0, 1))

    for ax in (ax_loss, ax_prob):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def train_dcgan(args, device, out_dir):
    """Train the DCGAN baseline with the standard alternating GAN updates."""
    train_loader, _, _ = get_dataloaders(args.root, args.batch_size,
                                         num_workers=0,
                                         max_slices=args.max_slices)
    if len(train_loader) == 0:
        raise ValueError("No full batch available: lower --batch-size.")

    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # The same noise every time, so sample grids show progress across epochs.
    fixed_noise = torch.randn(16, args.latent_dim, device=device)
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        generator.train()
        start = time.time()
        totals = {"d_loss": 0.0, "g_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}

        for real in train_loader:
            real = to_tanh_range(real.to(device))
            batch = real.size(0)
            real_labels = torch.ones(batch, device=device)
            fake_labels = torch.zeros(batch, device=device)

            fake = generator(torch.randn(batch, args.latent_dim, device=device))

            # Discriminator step: push real towards 1 and fake towards 0.
            # detach() stops this loss from updating the generator.
            real_logits = discriminator(real)
            fake_logits = discriminator(fake.detach())
            d_loss = (criterion(real_logits, real_labels)
                      + criterion(fake_logits, fake_labels))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # Generator step: make the updated discriminator call fakes real.
            g_loss = criterion(discriminator(fake), real_labels)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            totals["d_loss"] += d_loss.item()
            totals["g_loss"] += g_loss.item()
            totals["d_real"] += torch.sigmoid(real_logits).mean().item()
            totals["d_fake"] += torch.sigmoid(fake_logits).mean().item()

        row: dict[str, float] = {"epoch": epoch}
        row.update({key: value / len(train_loader) for key, value in totals.items()})
        history.append(row)
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"D loss {row['d_loss']:.3f}  G loss {row['g_loss']:.3f}  "
              f"D(x) {row['d_real']:.2f}  D(G(z)) {row['d_fake']:.2f}  "
              f"{time.time() - start:.0f}s", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                samples = to_unit_range(generator(fixed_noise))
            save_image(samples, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"),
                       nrow=4)
            torch.save({"generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "epoch": epoch,
                        "latent_dim": args.latent_dim},
                       os.path.join(out_dir, "dcgan.pt"))
            save_history(history, out_dir)

    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Outputs saved to", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Train a HipMRI generative model.")
    parser.add_argument("--model", choices=["dcgan"], default="dcgan")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=os.path.expanduser("~/hipmri_runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Save samples, weights and curves every N epochs.")
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Use only the first N slices per split (quick checks).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    print("Device:", device)

    if args.model == "dcgan":
        train_dcgan(args, device, out_dir)


if __name__ == "__main__":
    main()