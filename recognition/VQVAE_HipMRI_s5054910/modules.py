"""
modules.py - Model components for HipMRI image generation.

Holds the DCGAN baseline (Radford et al., 2015) and the WGAN-GP critic
(Gulrajani et al., 2017), sized for 1 x 128 x 256 slices. Both GANs share
the same generator. The VQ-VAE components will be added to this file.

Run this file directly to check output shapes and parameter counts:
    python3 modules.py
"""

import torch
from torch import nn

LATENT_DIM = 100


def init_weights(module):
    """DCGAN initialisation: conv weights from N(0, 0.02), batch norm from N(1, 0.02)."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def count_parameters(model):
    """Number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def up_block(in_channels, out_channels):
    """Double the height and width: transposed conv, batch norm, ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_block(in_channels, out_channels, norm="batch"):
    """Halve the height and width: strided conv, normalisation, LeakyReLU.

    norm="batch" is used by the DCGAN discriminator. norm="layer" is used by
    the WGAN-GP critic, because the gradient penalty is computed per image
    and batch norm would mix images within a batch.
    """
    if norm == "batch":
        normalisation = nn.BatchNorm2d(out_channels)
    else:
        normalisation = nn.GroupNorm(1, out_channels)  # layer norm over (C, H, W)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        normalisation,
        nn.LeakyReLU(0.2, inplace=True),
    )


class Generator(nn.Module):
    """Maps a noise vector of size latent_dim to a 1 x 128 x 256 image in [-1, 1]."""

    def __init__(self, latent_dim=LATENT_DIM, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # A (4, 8) kernel on a 1 x 1 input sets the 1:2 aspect ratio.
            nn.ConvTranspose2d(latent_dim, base * 16, (4, 8), 1, 0, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),            # 4 x 8
            up_block(base * 16, base * 8),    # 8 x 16
            up_block(base * 8, base * 4),     # 16 x 32
            up_block(base * 4, base * 2),     # 32 x 64
            up_block(base * 2, base),         # 64 x 128
            nn.ConvTranspose2d(base, 1, 4, 2, 1, bias=False),  # 128 x 256
            nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    """Maps a 1 x 128 x 256 image to one logit (positive means "looks real")."""

    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1, bias=False),   # 64 x 128
            nn.LeakyReLU(0.2, inplace=True),
            down_block(base, base * 2),                # 32 x 64
            down_block(base * 2, base * 4),            # 16 x 32
            down_block(base * 4, base * 8),            # 8 x 16
            down_block(base * 8, base * 16),           # 4 x 8
            nn.Conv2d(base * 16, 1, (4, 8), 1, 0, bias=False),  # 1 x 1
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x).view(-1)


class Critic(nn.Module):
    """WGAN-GP critic: maps a 1 x 128 x 256 image to an unbounded realness score.

    Same layout as the DCGAN discriminator, but with layer normalisation
    instead of batch norm (Gulrajani et al., 2017).
    """

    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1, bias=False),           # 64 x 128
            nn.LeakyReLU(0.2, inplace=True),
            down_block(base, base * 2, norm="layer"),          # 32 x 64
            down_block(base * 2, base * 4, norm="layer"),      # 16 x 32
            down_block(base * 4, base * 8, norm="layer"),      # 8 x 16
            down_block(base * 8, base * 16, norm="layer"),     # 4 x 8
            nn.Conv2d(base * 16, 1, (4, 8), 1, 0, bias=False),  # 1 x 1
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x).view(-1)


if __name__ == "__main__":
    generator, discriminator, critic = Generator(), Discriminator(), Critic()
    noise = torch.randn(4, LATENT_DIM)
    fake = generator(noise)
    print("Generator output:", tuple(fake.shape),
          f"range {fake.min().item():.2f} to {fake.max().item():.2f}")
    print("Discriminator output:", tuple(discriminator(fake).shape))
    print("Critic output:", tuple(critic(fake).shape))
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")
    print(f"Critic parameters: {count_parameters(critic):,}")