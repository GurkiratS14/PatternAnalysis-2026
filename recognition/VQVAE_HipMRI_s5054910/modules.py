"""
modules.py - Model components for HipMRI image generation.

Holds the DCGAN baseline (Radford et al., 2015), the WGAN-GP critic
(Gulrajani et al., 2017) and the VQ-VAE (van den Oord et al., 2017), sized
for 1 x 128 x 256 slices. Both GANs share the same generator.

Run this file directly to check output shapes and parameter counts:
    python3 modules.py
"""

import torch
import torch.nn.functional as F
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


class ResidualBlock(nn.Module):
    """x + conv3x3(relu(conv1x1(relu(x)))) as in the VQ-VAE paper."""

    def __init__(self, channels, hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, hidden, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """1 x 128 x 256 image -> embedding_dim x 32 x 64 continuous latent grid."""

    def __init__(self, hidden=128, res_hidden=32, num_res=2, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden // 2, 4, 2, 1), nn.ReLU(inplace=True),       # 64 x 128
            nn.Conv2d(hidden // 2, hidden, 4, 2, 1), nn.ReLU(inplace=True),  # 32 x 64
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            *[ResidualBlock(hidden, res_hidden) for _ in range(num_res)],
            nn.ReLU(),
            nn.Conv2d(hidden, embedding_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """embedding_dim x 32 x 64 quantised grid -> 1 x 128 x 256 image."""

    def __init__(self, hidden=128, res_hidden=32, num_res=2, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, 3, 1, 1),
            *[ResidualBlock(hidden, res_hidden) for _ in range(num_res)],
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, 2, 1), nn.ReLU(inplace=True),  # 64 x 128
            nn.ConvTranspose2d(hidden // 2, 1, 4, 2, 1),                              # 128 x 256
        )

    def forward(self, z):
        return self.net(z)


class VectorQuantizer(nn.Module):
    """Snaps each latent vector to its nearest codebook vector.

    Returns the quantised grid (with straight-through gradients), the
    codebook + commitment loss, and the chosen code index at each position.
    """

    usage: torch.Tensor

    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer("usage", torch.zeros(num_embeddings))

    def forward(self, z):
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        codes = self.codebook.weight
        distances = (flat.pow(2).sum(1, keepdim=True)
                     - 2 * flat @ codes.t()
                     + codes.pow(2).sum(1))
        indices = distances.argmin(1)
        if self.training:
            self.usage += torch.bincount(indices, minlength=self.num_embeddings)

        quantized = self.codebook(indices).view(b, h, w, d).permute(0, 3, 1, 2)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach()  # straight-through estimator
        return quantized, codebook_loss + self.beta * commitment_loss, indices.view(b, h, w)

    def lookup(self, indices):
        """(B, H, W) code indices -> (B, D, H, W) codebook vectors."""
        return self.codebook(indices).permute(0, 3, 1, 2)

    def usage_stats(self):
        """Codes used since the last reset, and the perplexity of their usage."""
        probs = self.usage / self.usage.sum().clamp(min=1)
        nonzero = probs[probs > 0]
        perplexity = torch.exp(-(nonzero * nonzero.log()).sum())
        return int((self.usage > 0).sum()), perplexity.item()

    @torch.no_grad()
    def reset_dead_codes(self, encodings):
        """Move unused codes onto random recent encoder outputs, then clear usage."""
        dead = (self.usage == 0).nonzero().squeeze(1)
        if len(dead) > 0:
            picks = torch.randint(0, encodings.size(0), (len(dead),), device=encodings.device)
            self.codebook.weight.data[dead] = encodings[picks]
        self.usage.zero_()
        return len(dead)


class VQVAE(nn.Module):
    """Encoder -> vector quantiser -> decoder."""

    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.config = {"num_embeddings": num_embeddings,
                       "embedding_dim": embedding_dim, "beta": beta}
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim=embedding_dim)

    def forward(self, x):
        quantized, vq_loss, indices = self.quantizer(self.encoder(x))
        return self.decoder(quantized), vq_loss, indices

    @torch.no_grad()
    def encode(self, x):
        """Images -> (B, 32, 64) code indices."""
        return self.quantizer(self.encoder(x))[2]

    @torch.no_grad()
    def decode_indices(self, indices):
        """(B, 32, 64) code indices -> images."""
        return self.decoder(self.quantizer.lookup(indices))


if __name__ == "__main__":
    vqvae = VQVAE()
    images = torch.rand(2, 1, 128, 256)
    recon, vq_loss, codes = vqvae(images)
    print("VQ-VAE reconstruction:", tuple(recon.shape), "codes:", tuple(codes.shape))
    print(f"VQ-VAE parameters: {count_parameters(vqvae):,}")

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