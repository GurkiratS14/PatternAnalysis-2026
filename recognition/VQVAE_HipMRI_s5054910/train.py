"""
train.py - Training script for HipMRI image generation models.

Trains the DCGAN baseline, the WGAN-GP or the VQ-VAE. Saves sample or
reconstruction grids, training curves and model weights to --out-dir, which
should sit outside the repository.

Usage:
    python3 train.py --model dcgan --epochs 100
    python3 train.py --model wgan_gp --epochs 500
    python3 train.py --model vqvae --epochs 100
    python3 train.py --model pixelcnn --epochs 60   (needs the trained VQ-VAE)
Quick check on a few slices:
    python3 train.py --model wgan_gp --epochs 2 --max-slices 64 --batch-size 16
"""

import argparse
import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from dataset import DEFAULT_ROOT, HipMRISlices, get_dataloaders
from metrics import ssim_pairs
from modules import (LATENT_DIM, VQVAE, Critic, Discriminator, Generator, PixelCNN,
                     count_parameters)

DEFAULT_LR = {"dcgan": 2e-4, "wgan_gp": 1e-4, "vqvae": 2e-4, "pixelcnn": 3e-4}

PIXELCNN_PANELS = [
    {"title": "Prior negative log-likelihood", "ylabel": "Bits per code",
     "series": [("train_bits", "Training"), ("val_bits", "Validation")]},
]

VQVAE_PANELS = [
    {"title": "Reconstruction error", "ylabel": "MSE",
     "series": [("recon_loss", "Training"), ("val_mse", "Validation")]},
    {"title": "Validation reconstruction SSIM", "ylabel": "Mean SSIM",
     "ylim": (0, 1), "hline": 0.6, "series": [("val_ssim", "Validation")]},
    {"title": "Codebook usage", "ylabel": "Codes used per epoch",
     "series": [("codes_used", "Codes used")]},
]

DCGAN_PANELS = [
    {"title": "Training loss", "ylabel": "BCE loss",
     "series": [("d_loss", "Discriminator"), ("g_loss", "Generator")]},
    {"title": "Discriminator's belief that an image is real",
     "ylabel": "Mean probability", "ylim": (0, 1), "hline": 0.5,
     "series": [("d_real", "D(x): real images"), ("d_fake", "D(G(z)): fake images")]},
]

WGAN_PANELS = [
    {"title": "Training loss", "ylabel": "Loss",
     "series": [("critic_loss", "Critic"), ("g_loss", "Generator")]},
    {"title": "Estimated Wasserstein distance", "ylabel": "Mean real score - mean fake score",
     "series": [("wasserstein", "Critic estimate")]},
]


def to_tanh_range(images):
    """[0, 1] -> [-1, 1], to match the generator's tanh output."""
    return images * 2.0 - 1.0


def to_unit_range(images):
    """[-1, 1] -> [0, 1], for saving and for metrics."""
    return (images + 1.0) / 2.0


def make_train_loader(args):
    """Training DataLoader, after the patient leakage check in get_dataloaders."""
    train_loader, _, _ = get_dataloaders(args.root, args.batch_size, num_workers=0,
                                         max_slices=args.max_slices)
    if len(train_loader) == 0:
        raise ValueError("No full batch available: lower --batch-size.")
    return train_loader


def save_history(history, out_dir, panels):
    """Write per-epoch statistics to a CSV file and plot them, one panel per group."""
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    colours, styles = ["#2a78d6", "#eb6834"], ["-", "--"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for i, (key, label) in enumerate(panel["series"]):
            ax.plot(epochs, [row[key] for row in history], color=colours[i],
                    linestyle=styles[i], linewidth=2, label=label)
        if "hline" in panel:
            ax.axhline(panel["hline"], color="gray", linewidth=1, linestyle=":")
        if "ylim" in panel:
            ax.set_ylim(*panel["ylim"])
        ax.set(title=panel["title"], xlabel="Epoch", ylabel=panel["ylabel"])
        ax.grid(alpha=0.3)
        if len(panel["series"]) > 1:
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def save_progress(generator, fixed_noise, epoch, out_dir, checkpoint):
    """Save a sample grid from the fixed noise, and the model weights."""
    generator.eval()
    with torch.no_grad():
        samples = to_unit_range(generator(fixed_noise))
    generator.train()
    save_image(samples, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"), nrow=4)
    name = os.path.basename(out_dir)
    torch.save(checkpoint, os.path.join(out_dir, f"{name}.pt"))


def print_epoch(epoch, total, row, seconds):
    """One line per epoch with every logged statistic."""
    stats = "  ".join(f"{key} {value:.3f}" for key, value in row.items() if key != "epoch")
    print(f"Epoch {epoch:3d}/{total}  {stats}  {seconds:.0f}s", flush=True)


def train_dcgan(args, device, out_dir):
    """Train the DCGAN baseline with the standard alternating GAN updates."""
    train_loader = make_train_loader(args)
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
        print_epoch(epoch, args.epochs, row, time.time() - start)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_progress(generator, fixed_noise, epoch, out_dir,
                          {"generator": generator.state_dict(),
                           "discriminator": discriminator.state_dict(),
                           "epoch": epoch, "latent_dim": args.latent_dim})
            save_history(history, out_dir, DCGAN_PANELS)


def gradient_penalty(critic, real, fake):
    """Mean of (||grad critic(x_hat)|| - 1)^2 at random points between real and fake."""
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grads, = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(args, device, out_dir):
    """Train the WGAN-GP: n_critic critic steps for every generator step."""
    train_loader = make_train_loader(args)
    generator = Generator(args.latent_dim).to(device)
    critic = Critic().to(device)
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Critic parameters: {count_parameters(critic):,}")

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(16, args.latent_dim, device=device)
    history = []
    critic_steps = 0  # counted across epochs, so the n_critic rhythm never resets

    for epoch in range(1, args.epochs + 1):
        generator.train()
        start = time.time()
        totals = {"critic_loss": 0.0, "wasserstein": 0.0, "gp": 0.0, "g_loss": 0.0}
        generator_steps = 0

        for real in train_loader:
            real = to_tanh_range(real.to(device))
            batch = real.size(0)

            # Critic step: score real high and fake low, with gradients kept near 1.
            with torch.no_grad():
                fake = generator(torch.randn(batch, args.latent_dim, device=device))
            real_score = critic(real).mean()
            fake_score = critic(fake).mean()
            penalty = gradient_penalty(critic, real, fake)
            critic_loss = fake_score - real_score + args.gp_weight * penalty
            opt_c.zero_grad(set_to_none=True)
            critic_loss.backward()
            opt_c.step()
            critic_steps += 1

            totals["critic_loss"] += critic_loss.item()
            totals["wasserstein"] += (real_score - fake_score).item()
            totals["gp"] += penalty.item()

            # Generator step every n_critic critic steps: raise the critic's score of fakes.
            if critic_steps % args.n_critic == 0:
                fake = generator(torch.randn(batch, args.latent_dim, device=device))
                g_loss = -critic(fake).mean()
                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_g.step()
                totals["g_loss"] += g_loss.item()
                generator_steps += 1

        row: dict[str, float] = {"epoch": epoch}
        for key in ("critic_loss", "wasserstein", "gp"):
            row[key] = totals[key] / len(train_loader)
        row["g_loss"] = totals["g_loss"] / generator_steps if generator_steps else float("nan")
        history.append(row)
        print_epoch(epoch, args.epochs, row, time.time() - start)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_progress(generator, fixed_noise, epoch, out_dir,
                          {"generator": generator.state_dict(),
                           "critic": critic.state_dict(),
                           "epoch": epoch, "latent_dim": args.latent_dim})
            save_history(history, out_dir, WGAN_PANELS)


@torch.no_grad()
def evaluate_reconstruction(model, loader, device):
    """Mean MSE and mean paired SSIM of reconstructions over a DataLoader."""
    model.eval()
    total_mse, total_ssim, count = 0.0, 0.0, 0
    for real in loader:
        real = real.to(device)
        recon = model(real)[0].clamp(0, 1)
        total_mse += F.mse_loss(recon, real, reduction="sum").item() / real[0].numel()
        total_ssim += ssim_pairs(recon, real).sum().item()
        count += real.size(0)
    model.train()
    return total_mse / count, total_ssim / count


def train_vqvae(args, device, out_dir):
    """Train the VQ-VAE on reconstruction, keeping the checkpoint with the best validation SSIM."""
    train_loader, val_loader, _ = get_dataloaders(args.root, args.batch_size, num_workers=0,
                                                  max_slices=args.max_slices)
    if len(train_loader) == 0:
        raise ValueError("No full batch available: lower --batch-size.")

    model = VQVAE(args.codebook_size, args.embedding_dim, args.beta).to(device)
    print(f"VQ-VAE parameters: {count_parameters(model):,}")
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_set = val_loader.dataset
    assert isinstance(val_set, HipMRISlices)
    picks = torch.linspace(0, len(val_set) - 1, 8).long().tolist()
    fixed = torch.stack([val_set[i] for i in picks]).to(device)
    history, best_ssim = [], -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        totals = {"recon_loss": 0.0, "vq_loss": 0.0}
        last_batch = torch.empty(0)

        for real in train_loader:
            real = real.to(device)
            last_batch = real
            recon, vq_loss, _ = model(real)
            recon_loss = F.mse_loss(recon, real)
            loss = recon_loss + vq_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            totals["recon_loss"] += recon_loss.item()
            totals["vq_loss"] += vq_loss.item()

        codes_used, perplexity = model.quantizer.usage_stats()
        with torch.no_grad():
            latents = model.encoder(last_batch)
            latents = latents.permute(0, 2, 3, 1).reshape(-1, latents.size(1))
        restarted = model.quantizer.reset_dead_codes(latents)
        val_mse, val_ssim = evaluate_reconstruction(model, val_loader, device)

        row: dict[str, float] = {"epoch": epoch}
        row.update({key: value / len(train_loader) for key, value in totals.items()})
        row.update({"val_mse": val_mse, "val_ssim": val_ssim, "codes_used": codes_used,
                    "perplexity": perplexity, "restarted": restarted})
        history.append(row)
        print_epoch(epoch, args.epochs, row, time.time() - start)

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({"model": model.state_dict(), "config": model.config,
                        "epoch": epoch, "val_ssim": val_ssim},
                       os.path.join(out_dir, "vqvae.pt"))

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                recon = model(fixed)[0].clamp(0, 1)
            model.train()
            save_image(torch.cat([fixed, recon]),
                       os.path.join(out_dir, f"recon_epoch_{epoch:03d}.png"), nrow=8)
            save_history(history, out_dir, VQVAE_PANELS)

    print(f"Best validation SSIM: {best_ssim:.3f}")


def load_vqvae(weights, device):
    """Load a trained VQ-VAE in evaluation mode."""
    checkpoint = torch.load(weights, map_location=device)
    vqvae = VQVAE(**checkpoint["config"]).to(device)
    vqvae.load_state_dict(checkpoint["model"])
    return vqvae.eval()


@torch.no_grad()
def encode_dataset(vqvae, dataset, device, batch_size=256):
    """Encode every slice of a dataset into its (32, 64) grid of code indices."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return torch.cat([vqvae.encode(batch.to(device)).cpu() for batch in loader])


@torch.no_grad()
def prior_loss(prior, codes, device, batch_size=64):
    """Mean cross-entropy (nats per code) of a prior on a set of code grids."""
    prior.eval()
    total = 0.0
    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size].to(device)
        total += F.cross_entropy(prior(batch), batch, reduction="sum").item()
    prior.train()
    return total / codes.numel()


def train_pixelcnn(args, device, out_dir):
    """Train the PixelCNN prior on the code grids of the trained VQ-VAE."""
    vqvae_weights = args.vqvae_weights or os.path.join(args.out_dir, "vqvae", "vqvae.pt")
    vqvae = load_vqvae(vqvae_weights, device)
    train_loader, val_loader, _ = get_dataloaders(args.root, args.batch_size, num_workers=0,
                                                  max_slices=args.max_slices)
    train_codes = encode_dataset(vqvae, train_loader.dataset, device)
    val_codes = encode_dataset(vqvae, val_loader.dataset, device)
    height, width = train_codes.shape[1], train_codes.shape[2]
    print(f"Encoded {len(train_codes)} training and {len(val_codes)} validation slices "
          f"into {height} x {width} code grids")

    code_loader = DataLoader(TensorDataset(train_codes), batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    if len(code_loader) == 0:
        raise ValueError("No full batch available: lower --batch-size.")

    prior = PixelCNN(vqvae.config["num_embeddings"], args.prior_dim, args.prior_layers,
                     dropout=args.prior_dropout).to(device)
    print(f"PixelCNN parameters: {count_parameters(prior):,} "
          f"(dropout {args.prior_dropout}, weight decay {args.weight_decay})")
    # AdamW applies weight decay directly to the weights (Loshchilov and Hutter, 2019).
    optimiser = torch.optim.AdamW(prior.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    history, best_loss = [], float("inf")

    for epoch in range(1, args.epochs + 1):
        prior.train()
        start = time.time()
        total = 0.0
        for (codes,) in code_loader:
            codes = codes.to(device)
            loss = F.cross_entropy(prior(codes), codes)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item()

        val_loss = prior_loss(prior, val_codes, device)
        row: dict[str, float] = {"epoch": epoch,
                                 "train_bits": total / len(code_loader) / math.log(2),
                                 "val_bits": val_loss / math.log(2)}
        history.append(row)
        print_epoch(epoch, args.epochs, row, time.time() - start)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": prior.state_dict(), "config": prior.config,
                        "grid": (height, width),
                        "vqvae_weights": vqvae_weights, "epoch": epoch,
                        "val_bits": val_loss / math.log(2)},
                       os.path.join(out_dir, "pixelcnn.pt"))

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sampled = prior.sample(8, height, width, device)
            with torch.no_grad():
                images = vqvae.decode_indices(sampled).clamp(0, 1)
            save_image(images, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"), nrow=4)
            save_history(history, out_dir, PIXELCNN_PANELS)

    print(f"Best validation: {best_loss / math.log(2):.3f} bits per code")


def main():
    parser = argparse.ArgumentParser(description="Train a HipMRI generative model.")
    parser.add_argument("--model", choices=["dcgan", "wgan_gp", "vqvae", "pixelcnn"],
                        default="dcgan")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=os.path.expanduser("~/hipmri_runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to 2e-4 for dcgan and 1e-4 for wgan_gp.")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--n-critic", type=int, default=5,
                        help="WGAN-GP: critic steps per generator step.")
    parser.add_argument("--gp-weight", type=float, default=10.0,
                        help="WGAN-GP: weight of the gradient penalty.")
    parser.add_argument("--codebook-size", type=int, default=512,
                        help="VQ-VAE: number of codebook vectors.")
    parser.add_argument("--embedding-dim", type=int, default=64,
                        help="VQ-VAE: length of each codebook vector.")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="VQ-VAE: weight of the commitment loss.")
    parser.add_argument("--vqvae-weights", default=None,
                        help="PixelCNN: trained VQ-VAE, default <out-dir>/vqvae/vqvae.pt.")
    parser.add_argument("--prior-dim", type=int, default=128,
                        help="PixelCNN: channels per layer.")
    parser.add_argument("--prior-dropout", type=float, default=0.0,
                        help="PixelCNN: dropout probability in each gated layer.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="PixelCNN: AdamW weight decay.")
    parser.add_argument("--run-name", default=None,
                        help="Output sub-folder, default the model name (e.g. pixelcnn_reg).")
    parser.add_argument("--prior-layers", type=int, default=12,
                        help="PixelCNN: number of gated layers.")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Save samples, weights and curves every N epochs.")
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Use only the first N slices per split (quick checks).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LR[args.model]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_dir, args.run_name or args.model)
    os.makedirs(out_dir, exist_ok=True)
    print("Device:", device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if args.model == "dcgan":
        train_dcgan(args, device, out_dir)
    elif args.model == "wgan_gp":
        train_wgan_gp(args, device, out_dir)
    elif args.model == "vqvae":
        train_vqvae(args, device, out_dir)
    else:
        train_pixelcnn(args, device, out_dir)

    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Outputs saved to", out_dir)


if __name__ == "__main__":
    main()