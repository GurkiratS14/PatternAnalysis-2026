# --------------------------------------------------------------------
# File:    train.py
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Training driver for DCGAN, WGAN-GP, VQ-VAE, VQ-VAE-2 and the
#          PixelCNN priors on HipMRI 2D slices.
# --------------------------------------------------------------------
"""
train.py - Training script for HipMRI image generation models.

Trains the DCGAN and WGAN-GP baselines, the VQ-VAE and VQ-VAE-2, and the
PixelCNN priors over their codes. Saves sample or
reconstruction grids, training curves and model weights to --out-dir, which
should sit outside the repository.

Usage:
    python3 train.py --model dcgan --epochs 100
    python3 train.py --model wgan_gp --epochs 500
    python3 train.py --model vqvae --epochs 100
    python3 train.py --model pixelcnn --epochs 60   (needs the trained VQ-VAE)
    python3 train.py --model vqvae2 --epochs 100
    python3 train.py --model pixelcnn --level top --run-name pixelcnn2_top \
        --vqvae-weights ~/hipmri_runs/vqvae2/vqvae2.pt
    python3 train.py --model pixelcnn --level bottom --run-name pixelcnn2 \
        --vqvae-weights ~/hipmri_runs/vqvae2/vqvae2.pt
With flip augmentation, early stopping and (for priors) gradient clipping:
    python3 train.py --model vqvae2 --augment --patience 10 --run-name vqvae2_aug
    python3 train.py --model pixelcnn --level bottom --augment --grad-clip 1.0 --patience 10 \
        --vqvae-weights ~/hipmri_runs/vqvae2_aug/vqvae2.pt --run-name pixelcnn2_aug
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
from modules import (LATENT_DIM, VQVAE, VQVAE2, Critic, Discriminator, Generator, PixelCNN,
                     count_parameters)

# Default learning rates: the DCGAN paper's 2e-4, the WGAN-GP paper's 1e-4.
DEFAULT_LR = {"dcgan": 2e-4, "wgan_gp": 1e-4, "vqvae": 2e-4, "vqvae2": 2e-4,
              "pixelcnn": 3e-4}

# Training-curve layouts: one dict per panel, listing the history keys to plot.
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

VQVAE2_PANELS = VQVAE_PANELS[:2] + [
    {"title": "Codebook usage", "ylabel": "Codes used per epoch",
     "series": [("codes_used", "Bottom"), ("top_codes_used", "Top")]},
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


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def to_tanh_range(images):
    """[0, 1] -> [-1, 1], to match the generator's tanh output."""
    return images * 2.0 - 1.0


def to_unit_range(images):
    """[-1, 1] -> [0, 1], for saving and for metrics."""
    return (images + 1.0) / 2.0


def make_train_loader(args):
    """Training DataLoader, after the patient leakage check in get_dataloaders."""
    # --augment flips training slices left-right at random; validation is never flipped.
    train_loader, _, _ = get_dataloaders(args.root, args.batch_size, num_workers=0,
                                         augment=args.augment, max_slices=args.max_slices)
    # drop_last=True means a batch size larger than the split gives no batches at all.
    if len(train_loader) == 0:
        raise ValueError("No full batch available: lower --batch-size.")
    return train_loader


def accumulate(totals, stats):
    """Add one step's statistics to the running totals (keeps first-seen key order)."""
    for key, value in stats.items():
        totals[key] = totals.get(key, 0.0) + value


def average(totals, count):
    """Divide every running total by the number of steps."""
    return {key: value / count for key, value in totals.items()}


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
        # Optional reference line, e.g. the 0.6 SSIM target.
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
    # eval() so batch norm uses its running statistics, as it will at inference time.
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


def is_save_epoch(epoch, args):
    """True every --sample-every epochs and on the final epoch."""
    return epoch % args.sample_every == 0 or epoch == args.epochs


def should_stop(epoch, best_epoch, patience):
    """Early stopping: True once validation has not improved for `patience` epochs."""
    return patience > 0 and epoch - best_epoch >= patience


# ----------------------------------------------------------------------------
# DCGAN baseline
# ----------------------------------------------------------------------------

def dcgan_step(generator, discriminator, real, criterion, opt_g, opt_d, latent_dim):
    """One discriminator update followed by one generator update on a batch in [-1, 1]."""
    batch = real.size(0)
    real_labels = torch.ones(batch, device=real.device)
    fake_labels = torch.zeros(batch, device=real.device)
    fake = generator(torch.randn(batch, latent_dim, device=real.device))

    # Discriminator: push real towards 1 and fake towards 0.
    # detach() stops this loss from updating the generator.
    real_logits = discriminator(real)
    fake_logits = discriminator(fake.detach())
    d_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)
    opt_d.zero_grad(set_to_none=True)
    d_loss.backward()
    opt_d.step()

    # Generator: make the just-updated discriminator call the same fakes real
    # (the non-saturating loss, which gives stronger gradients early in training).
    g_loss = criterion(discriminator(fake), real_labels)
    opt_g.zero_grad(set_to_none=True)
    g_loss.backward()
    opt_g.step()

    # D(x) and D(G(z)) as probabilities show which network is winning.
    return {"d_loss": d_loss.item(), "g_loss": g_loss.item(),
            "d_real": torch.sigmoid(real_logits).mean().item(),
            "d_fake": torch.sigmoid(fake_logits).mean().item()}


def train_dcgan(args, device, out_dir):
    """Train the DCGAN baseline with the standard alternating GAN updates."""
    train_loader = make_train_loader(args)
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")

    # BCE on logits is numerically safer than a sigmoid followed by BCE.
    criterion = nn.BCEWithLogitsLoss()
    # beta1 = 0.5, as in the DCGAN paper: less momentum keeps the two players stable.
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # The same noise every time, so sample grids show progress across epochs.
    fixed_noise = torch.randn(16, args.latent_dim, device=device)
    history = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        start = time.time()
        totals = {}
        for real in train_loader:
            # The generator ends in tanh, so real images are moved to [-1, 1] too.
            stats = dcgan_step(generator, discriminator, to_tanh_range(real.to(device)),
                               criterion, opt_g, opt_d, args.latent_dim)
            accumulate(totals, stats)

        row: dict[str, float] = {"epoch": epoch, **average(totals, len(train_loader))}
        history.append(row)
        print_epoch(epoch, args.epochs, row, time.time() - start)

        if is_save_epoch(epoch, args):
            save_progress(generator, fixed_noise, epoch, out_dir,
                          {"generator": generator.state_dict(),
                           "discriminator": discriminator.state_dict(),
                           "epoch": epoch, "latent_dim": args.latent_dim})
            save_history(history, out_dir, DCGAN_PANELS)


# ----------------------------------------------------------------------------
# WGAN-GP baseline
# ----------------------------------------------------------------------------

def gradient_penalty(critic, real, fake):
    """Mean of (||grad critic(x_hat)|| - 1)^2 at random points between real and fake."""
    # One random mixing weight per image; x_hat lies on the line between a real and a fake.
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)
    # create_graph=True so the penalty itself can be backpropagated into the critic.
    grads, = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def critic_step(generator, critic, real, opt_c, args):
    """One critic update: score real high and fake low, with gradient norms kept near 1."""
    # The generator is not being trained here, so no graph is built through it.
    with torch.no_grad():
        fake = generator(torch.randn(real.size(0), args.latent_dim, device=real.device))
    real_score = critic(real).mean()
    fake_score = critic(fake).mean()
    penalty = gradient_penalty(critic, real, fake)
    critic_loss = fake_score - real_score + args.gp_weight * penalty
    opt_c.zero_grad(set_to_none=True)
    critic_loss.backward()
    opt_c.step()
    # real - fake is the critic's estimate of the Wasserstein distance.
    return {"critic_loss": critic_loss.item(),
            "wasserstein": (real_score - fake_score).item(),
            "gp": penalty.item()}


def generator_step(generator, critic, batch, opt_g, args, device):
    """One generator update: raise the critic's score of freshly generated images."""
    fake = generator(torch.randn(batch, args.latent_dim, device=device))
    g_loss = -critic(fake).mean()
    opt_g.zero_grad(set_to_none=True)
    g_loss.backward()
    opt_g.step()
    return g_loss.item()


def train_wgan_gp(args, device, out_dir):
    """Train the WGAN-GP: n_critic critic steps for every generator step."""
    train_loader = make_train_loader(args)
    generator = Generator(args.latent_dim).to(device)
    critic = Critic().to(device)
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Critic parameters: {count_parameters(critic):,}")

    # Adam(0, 0.9) as recommended by Gulrajani et al. (2017) for WGAN-GP.
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(16, args.latent_dim, device=device)
    history = []
    critic_steps = 0  # counted across epochs, so the n_critic rhythm never resets

    for epoch in range(1, args.epochs + 1):
        generator.train()
        start = time.time()
        totals, g_total, generator_steps = {}, 0.0, 0

        for real in train_loader:
            real = to_tanh_range(real.to(device))
            accumulate(totals, critic_step(generator, critic, real, opt_c, args))
            critic_steps += 1
            # The critic trains n_critic times as often, so its estimate stays accurate.
            if critic_steps % args.n_critic == 0:
                g_total += generator_step(generator, critic, real.size(0), opt_g, args, device)
                generator_steps += 1

        row: dict[str, float] = {"epoch": epoch, **average(totals, len(train_loader))}
        # The generator loss is averaged over generator steps only, not over all batches.
        row["g_loss"] = g_total / generator_steps if generator_steps else float("nan")
        history.append(row)
        print_epoch(epoch, args.epochs, row, time.time() - start)

        if is_save_epoch(epoch, args):
            save_progress(generator, fixed_noise, epoch, out_dir,
                          {"generator": generator.state_dict(),
                           "critic": critic.state_dict(),
                           "epoch": epoch, "latent_dim": args.latent_dim})
            save_history(history, out_dir, WGAN_PANELS)


# ----------------------------------------------------------------------------
# VQ-VAE and VQ-VAE-2
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_reconstruction(model, loader, device):
    """Mean MSE and mean paired SSIM of reconstructions over a DataLoader."""
    model.eval()
    total_mse, total_ssim, count = 0.0, 0.0, 0
    for real in loader:
        real = real.to(device)
        # Clamp to the valid intensity range before scoring, as the saved images are.
        recon = model(real)[0].clamp(0, 1)
        total_mse += F.mse_loss(recon, real, reduction="sum").item() / real[0].numel()
        total_ssim += ssim_pairs(recon, real).sum().item()
        count += real.size(0)
    model.train()
    return total_mse / count, total_ssim / count


def build_autoencoder(kind, config):
    """Rebuild a VQ-VAE or VQ-VAE-2 from its saved configuration."""
    return (VQVAE2 if kind == "vqvae2" else VQVAE)(**config)


def vqvae_epoch(model, loader, optimiser, device):
    """One pass over the training slices. Returns mean losses and the last batch seen."""
    totals, last_batch = {}, torch.empty(0)
    for real in loader:
        real = real.to(device)
        last_batch = real
        recon, vq_loss, _ = model(real)
        # vq_loss already holds the codebook term and beta x the commitment term.
        recon_loss = F.mse_loss(recon, real)
        loss = recon_loss + vq_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        accumulate(totals, {"recon_loss": recon_loss.item(), "vq_loss": vq_loss.item()})
    return average(totals, len(loader)), last_batch


def maintain_codebooks(model, batch):
    """Log codebook usage and restart dead codes, for each quantiser (one, or top and bottom)."""
    stats = {}
    latents = model.pre_quant_latents(batch)
    for (name, quantizer), z in zip(model.quantizers(), latents):
        prefix = f"{name}_" if name == "top" else ""
        used, perplexity = quantizer.usage_stats()
        # Unused codes are moved onto recent encoder outputs so they start being chosen.
        flat = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))
        stats.update({f"{prefix}codes_used": used, f"{prefix}perplexity": perplexity,
                      f"{prefix}restarted": quantizer.reset_dead_codes(flat)})
    return stats


def save_reconstructions(model, fixed, epoch, out_dir):
    """Save the fixed validation slices (top row) above their reconstructions."""
    model.eval()
    with torch.no_grad():
        recon = model(fixed)[0].clamp(0, 1)
    model.train()
    save_image(torch.cat([fixed, recon]),
               os.path.join(out_dir, f"recon_epoch_{epoch:03d}.png"), nrow=8)


def train_vqvae(args, device, out_dir):
    """Train a VQ-VAE or VQ-VAE-2 on reconstruction, keeping the best validation-SSIM checkpoint."""
    train_loader, val_loader, _ = get_dataloaders(args.root, args.batch_size, num_workers=0,
                                                  augment=args.augment,
                                                  max_slices=args.max_slices)
    if len(train_loader) == 0:
        raise ValueError("No full batch available: lower --batch-size.")

    model = build_autoencoder(args.model, {"num_embeddings": args.codebook_size,
                                           "embedding_dim": args.embedding_dim,
                                           "beta": args.beta}).to(device)
    print(f"{args.model} parameters: {count_parameters(model):,}")
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Eight evenly spaced validation slices, reconstructed at every save to show progress.
    val_set = val_loader.dataset
    assert isinstance(val_set, HipMRISlices)
    picks = torch.linspace(0, len(val_set) - 1, 8).long().tolist()
    fixed = torch.stack([val_set[i] for i in picks]).to(device)
    history, best_ssim, best_epoch = [], -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        losses, last_batch = vqvae_epoch(model, train_loader, optimiser, device)
        stats = maintain_codebooks(model, last_batch)
        val_mse, val_ssim = evaluate_reconstruction(model, val_loader, device)

        row: dict[str, float] = {"epoch": epoch, **losses,
                                 "val_mse": val_mse, "val_ssim": val_ssim, **stats}
        history.append(row)
        print_epoch(epoch, args.epochs, row, time.time() - start)

        # Keep the checkpoint with the best validation SSIM, never simply the last one.
        if val_ssim > best_ssim:
            best_ssim, best_epoch = val_ssim, epoch
            torch.save({"model": model.state_dict(), "config": model.config, "kind": args.model,
                        "epoch": epoch, "val_ssim": val_ssim},
                       os.path.join(out_dir, f"{args.model}.pt"))

        stop = should_stop(epoch, best_epoch, args.patience)
        if is_save_epoch(epoch, args) or stop:
            save_reconstructions(model, fixed, epoch, out_dir)
            save_history(history, out_dir,
                         VQVAE2_PANELS if args.model == "vqvae2" else VQVAE_PANELS)
        if stop:
            print(f"Early stopping at epoch {epoch}: no improvement for {args.patience} epochs")
            break

    print(f"Best validation SSIM: {best_ssim:.3f} (epoch {best_epoch})")


# ----------------------------------------------------------------------------
# PixelCNN priors over the code grids
# ----------------------------------------------------------------------------

def load_vqvae(weights, device):
    """Load a trained VQ-VAE or VQ-VAE-2 in evaluation mode, and return it with its kind."""
    checkpoint = torch.load(weights, map_location=device)
    # Checkpoints saved before VQ-VAE-2 existed have no "kind" and are single-level.
    kind = checkpoint.get("kind", "vqvae")
    vqvae = build_autoencoder(kind, checkpoint["config"]).to(device)
    vqvae.load_state_dict(checkpoint["model"])
    return vqvae.eval(), kind


@torch.no_grad()
def encode_dataset(vqvae, dataset, device, batch_size=256):
    """Encode every slice into its code grids: one grid per level, each (N, H, W)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    levels = [vqvae.encode_levels(batch.to(device)) for batch in loader]
    return [torch.cat([batch[i].cpu() for batch in levels]) for i in range(len(levels[0]))]


def load_code_grids(args, vqvae, level, device):
    """Code grids to model (train, val) and, for the bottom level, the top grids it is
    conditioned on (train, val; None otherwise)."""
    train_loader, val_loader, _ = get_dataloaders(args.root, args.batch_size, num_workers=0,
                                                  max_slices=args.max_slices)
    # The autoencoder is frozen, so every slice is encoded once, up front.
    train_set = train_loader.dataset
    train_levels = encode_dataset(vqvae, train_set, device)
    val_levels = encode_dataset(vqvae, val_loader.dataset, device)
    if args.augment:
        # Augmentation for the prior: also encode every training slice mirrored
        # left-right, doubling the code grids it learns from (validation is untouched).
        assert isinstance(train_set, HipMRISlices) and train_set.images is not None
        flipped = encode_dataset(vqvae, train_set.images.flip(-1), device)
        train_levels = [torch.cat([codes, extra]) for codes, extra in zip(train_levels, flipped)]
    # Levels are ordered [top, bottom] for VQ-VAE-2 and [single] for VQ-VAE.
    index = 0 if level == "top" else -1
    train_cond = train_levels[0] if level == "bottom" else None
    val_cond = val_levels[0] if level == "bottom" else None
    return train_levels[index], val_levels[index], train_cond, val_cond


@torch.no_grad()
def prior_loss(prior, codes, cond, device, batch_size=64):
    """Mean cross-entropy (nats per code) of a prior on a set of code grids."""
    prior.eval()  # no dropout when measuring validation loss
    total = 0.0
    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size].to(device)
        context = cond[start:start + batch_size].to(device) if cond is not None else None
        total += F.cross_entropy(prior(batch, context), batch, reduction="sum").item()
    prior.train()
    return total / codes.numel()


def prior_epoch(prior, code_loader, optimiser, device, grad_clip=0.0):
    """One pass over the training code grids. Returns the mean loss in nats per code."""
    total = 0.0
    for batch in code_loader:
        codes = batch[0].to(device)
        cond = batch[1].to(device) if len(batch) > 1 else None
        # The target at every position is the code itself; the masks ensure the
        # logits there were computed only from the codes before it.
        loss = F.cross_entropy(prior(codes, cond), codes)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            # Caps the gradient norm, so one unusual batch cannot destabilise training.
            torch.nn.utils.clip_grad_norm_(prior.parameters(), grad_clip)
        optimiser.step()
        total += loss.item()
    return total / len(code_loader)


def save_prior_samples(prior, vqvae, level, val_cond, grid, epoch, out_dir, device):
    """Decode a few sampled code grids so training progress can be seen."""
    height, width = grid
    path = os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png")
    if level == "single":
        sampled = prior.sample(8, height, width, device)
        with torch.no_grad():
            images = vqvae.decode_levels([sampled]).clamp(0, 1)
        save_image(images, path, nrow=4)
    elif level == "bottom" and val_cond is not None:
        # New bottom detail generated for 8 real validation layouts.
        top = val_cond[torch.linspace(0, len(val_cond) - 1, 8).long()].to(device)
        sampled = prior.sample(8, height, width, device, cond_codes=top)
        with torch.no_grad():
            images = vqvae.decode_levels([top, sampled]).clamp(0, 1)
        save_image(images, path, nrow=4)
    # A top-level grid cannot be decoded on its own, so no images are saved for it.


def train_pixelcnn(args, device, out_dir):
    """Train a PixelCNN prior on the code grids of a trained VQ-VAE or VQ-VAE-2.

    For VQ-VAE-2, --level top trains an unconditional prior over the 16 x 32 top
    grids and --level bottom trains a prior over the 32 x 64 bottom grids that is
    conditioned on the top grids.
    """
    vqvae_weights = args.vqvae_weights or os.path.join(args.out_dir, "vqvae", "vqvae.pt")
    vqvae, kind = load_vqvae(vqvae_weights, device)
    if kind == "vqvae2" and args.level not in ("top", "bottom"):
        raise ValueError("A VQ-VAE-2 needs --level top or --level bottom.")
    level = args.level if kind == "vqvae2" else "single"

    train_codes, val_codes, train_cond, val_cond = load_code_grids(args, vqvae, level, device)
    grid = (train_codes.shape[1], train_codes.shape[2])
    print(f"Encoded {len(train_codes)} training and {len(val_codes)} validation slices "
          f"into {grid[0]} x {grid[1]} code grids ({level} level)")

    # The conditioning grids travel with their code grids, so shuffling keeps pairs aligned.
    tensors = (train_codes,) if train_cond is None else (train_codes, train_cond)
    code_loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    if len(code_loader) == 0:
        raise ValueError("No full batch available: lower --batch-size.")

    num_embeddings = vqvae.config["num_embeddings"]
    prior = PixelCNN(num_embeddings, args.prior_dim, args.prior_layers,
                     dropout=args.prior_dropout,
                     cond_embeddings=num_embeddings if level == "bottom" else 0).to(device)
    print(f"PixelCNN parameters: {count_parameters(prior):,} "
          f"(dropout {args.prior_dropout}, weight decay {args.weight_decay})")
    # AdamW applies weight decay directly to the weights (Loshchilov and Hutter, 2019).
    optimiser = torch.optim.AdamW(prior.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    history, best_loss, best_epoch = [], float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        prior.train()
        start = time.time()
        train_loss = prior_epoch(prior, code_loader, optimiser, device, args.grad_clip)
        val_loss = prior_loss(prior, val_codes, val_cond, device)
        # Reported in bits per code (nats / ln 2): log2(512) = 9 bits would be pure guessing.
        row: dict[str, float] = {"epoch": epoch, "train_bits": train_loss / math.log(2),
                                 "val_bits": val_loss / math.log(2)}
        history.append(row)
        print_epoch(epoch, args.epochs, row, time.time() - start)

        # The prior overfits quickly, so only the best-validation epoch is kept.
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save({"model": prior.state_dict(), "config": prior.config,
                        "grid": grid, "level": level,
                        "vqvae_weights": vqvae_weights, "epoch": epoch,
                        "val_bits": val_loss / math.log(2)},
                       os.path.join(out_dir, "pixelcnn.pt"))

        stop = should_stop(epoch, best_epoch, args.patience)
        if is_save_epoch(epoch, args) or stop:
            save_prior_samples(prior, vqvae, level, val_cond, grid, epoch, out_dir, device)
            save_history(history, out_dir, PIXELCNN_PANELS)
        if stop:
            print(f"Early stopping at epoch {epoch}: no improvement for {args.patience} epochs")
            break

    print(f"Best validation: {best_loss / math.log(2):.3f} bits per code (epoch {best_epoch})")


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def parse_args():
    """Command-line options for every model, with per-model learning-rate defaults."""
    parser = argparse.ArgumentParser(description="Train a HipMRI generative model.")
    parser.add_argument("--model", choices=["dcgan", "wgan_gp", "vqvae", "vqvae2", "pixelcnn"],
                        default="dcgan")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=os.path.expanduser("~/hipmri_runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to 2e-4 (dcgan, vqvae, vqvae2), 1e-4 (wgan_gp), "
                             "3e-4 (pixelcnn).")
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
    parser.add_argument("--level", choices=["single", "top", "bottom"], default="single",
                        help="PixelCNN: which VQ-VAE-2 level to model (single for a VQ-VAE).")
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
    parser.add_argument("--augment", action="store_true",
                        help="Random left-right flips of training slices (for priors: train "
                             "on the codes of original and mirrored slices).")
    parser.add_argument("--patience", type=int, default=0,
                        help="VQ-VAE and PixelCNN: stop after this many epochs without "
                             "validation improvement (0 = train for all --epochs).")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="PixelCNN: clip the gradient norm to this value (0 = off).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LR[args.model]
    return args


TRAINERS = {"dcgan": train_dcgan, "wgan_gp": train_wgan_gp, "vqvae": train_vqvae,
            "vqvae2": train_vqvae, "pixelcnn": train_pixelcnn}


def main():
    """Parse command-line arguments and train the requested model."""
    args = parse_args()
    # Fixed seed so a rerun gives the same initial weights, noise and batch order.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_dir, args.run_name or args.model)
    os.makedirs(out_dir, exist_ok=True)
    print("Device:", device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    TRAINERS[args.model](args, device, out_dir)

    # Peak memory is reported for the resource-profiling table in the README.
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Outputs saved to", out_dir)


if __name__ == "__main__":
    main()
