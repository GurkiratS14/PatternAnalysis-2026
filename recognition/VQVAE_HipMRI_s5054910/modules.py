"""
modules.py - Model components for HipMRI image generation.

Holds the DCGAN baseline (Radford et al., 2015), the WGAN-GP critic
(Gulrajani et al., 2017), the VQ-VAE (van den Oord et al., 2017), the
two-level VQ-VAE-2 (Razavi et al., 2019) and the gated PixelCNN prior over
their codes (van den Oord et al., 2016), sized for 1 x 128 x 256 slices.
Both GANs share the same generator.

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

    # Interface shared with VQVAE2, so training and evaluation code handles both.
    def quantizers(self):
        return [("", self.quantizer)]

    @torch.no_grad()
    def pre_quant_latents(self, x):
        return [self.encoder(x)]

    @torch.no_grad()
    def encode_levels(self, x):
        return [self.encode(x)]

    @torch.no_grad()
    def decode_levels(self, levels):
        return self.decode_indices(levels[-1])

    @torch.no_grad()
    def interpolate(self, pair, weights):
        """Blend two images' encoder outputs, re-quantise, decode. Returns images and codes."""
        latents = self.encoder(pair)
        blended = (1 - weights) * latents[0:1] + weights * latents[1:2]
        quantized, _, codes = self.quantizer(blended)
        return self.decoder(quantized), codes


class VQVAE2(nn.Module):
    """Two-level VQ-VAE (Razavi et al., 2019).

    A 16 x 32 top code grid captures global layout and a 32 x 64 bottom grid,
    computed with the top level as context, captures local detail. The
    decoder uses both. Each top code covers an 8 x 8 pixel patch, each bottom
    code a 4 x 4 patch.
    """

    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25, hidden=128,
                 res_hidden=32):
        super().__init__()
        self.config = {"num_embeddings": num_embeddings, "embedding_dim": embedding_dim,
                       "beta": beta, "hidden": hidden, "res_hidden": res_hidden}
        self.enc_b = nn.Sequential(
            nn.Conv2d(1, hidden // 2, 4, 2, 1), nn.ReLU(inplace=True),       # 64 x 128
            nn.Conv2d(hidden // 2, hidden, 4, 2, 1), nn.ReLU(inplace=True),  # 32 x 64
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            ResidualBlock(hidden, res_hidden), ResidualBlock(hidden, res_hidden), nn.ReLU(),
        )
        self.enc_t = nn.Sequential(
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.ReLU(inplace=True),       # 16 x 32
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            ResidualBlock(hidden, res_hidden), ResidualBlock(hidden, res_hidden), nn.ReLU(),
        )
        self.pre_t = nn.Conv2d(hidden, embedding_dim, 1)
        self.quantizer_t = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.dec_t = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, 3, 1, 1),
            ResidualBlock(hidden, res_hidden), ResidualBlock(hidden, res_hidden), nn.ReLU(),
            nn.ConvTranspose2d(hidden, embedding_dim, 4, 2, 1),               # 32 x 64
        )
        self.pre_b = nn.Conv2d(hidden + embedding_dim, embedding_dim, 1)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, beta)  # bottom level
        self.up_t = nn.ConvTranspose2d(embedding_dim, embedding_dim, 4, 2, 1)
        self.decoder = Decoder(embedding_dim=2 * embedding_dim)

    def _encode(self, x):
        h_b = self.enc_b(x)
        return self._from_bottom_features(h_b)

    def _from_bottom_features(self, h_b):
        z_t = self.pre_t(self.enc_t(h_b))
        q_t, loss_t, codes_t = self.quantizer_t(z_t)
        z_b = self.pre_b(torch.cat([h_b, self.dec_t(q_t)], dim=1))
        return z_t, q_t, loss_t, codes_t, z_b

    def _decode(self, q_t, q_b):
        return self.decoder(torch.cat([self.up_t(q_t), q_b], dim=1))

    def forward(self, x):
        _, q_t, loss_t, _, z_b = self._encode(x)
        q_b, loss_b, codes_b = self.quantizer(z_b)
        return self._decode(q_t, q_b), loss_t + loss_b, codes_b

    def quantizers(self):
        return [("top", self.quantizer_t), ("bottom", self.quantizer)]

    @torch.no_grad()
    def pre_quant_latents(self, x):
        was_training = self.training
        self.eval()  # do not count these passes in codebook usage
        z_t, _, _, _, z_b = self._encode(x)
        self.train(was_training)
        return [z_t, z_b]

    @torch.no_grad()
    def encode_levels(self, x):
        """Images -> [(B, 16, 32) top codes, (B, 32, 64) bottom codes]."""
        _, _, _, codes_t, z_b = self._encode(x)
        return [codes_t, self.quantizer(z_b)[2]]

    @torch.no_grad()
    def decode_levels(self, levels):
        codes_t, codes_b = levels
        return self._decode(self.quantizer_t.lookup(codes_t), self.quantizer.lookup(codes_b))

    @torch.no_grad()
    def interpolate(self, pair, weights):
        """Blend two images' bottom encoder features, then quantise both levels and decode."""
        features = self.enc_b(pair)
        blended = (1 - weights) * features[0:1] + weights * features[1:2]
        _, q_t, _, _, z_b = self._from_bottom_features(blended)
        q_b, _, codes_b = self.quantizer(z_b)
        return self._decode(q_t, q_b), codes_b


def gate(x):
    """Gated activation: tanh(first half) * sigmoid(second half) along channels."""
    a, b = x.chunk(2, dim=1)
    return torch.tanh(a) * torch.sigmoid(b)


class GatedLayer(nn.Module):
    """One gated PixelCNN layer with vertical and horizontal stacks (van den Oord et al., 2016).

    The vertical stack sees rows above the current position and the
    horizontal stack sees positions to its left in the same row. In the
    first layer the current position itself is also hidden, so no output
    ever depends on the code it is predicting.
    """

    v_mask: torch.Tensor
    h_mask: torch.Tensor

    def __init__(self, dim, kernel=7, first=False, dropout=0.0, conditional=False):
        super().__init__()
        half = kernel // 2
        self.first = first
        self.dropout = nn.Dropout(dropout)
        # Optional conditioning (e.g. on VQ-VAE-2 top-level codes), added before each gate.
        self.cond_v = nn.Conv2d(dim, 2 * dim, 1) if conditional else None
        self.cond_h = nn.Conv2d(dim, 2 * dim, 1) if conditional else None
        self.vertical = nn.Conv2d(dim, 2 * dim, (half + 1, kernel), padding=(half, half))
        self.horizontal = nn.Conv2d(dim, 2 * dim, (1, half + 1), padding=(0, half))
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.horiz_out = nn.Conv2d(dim, dim, 1)
        v_mask = torch.ones(1, 1, half + 1, kernel)
        h_mask = torch.ones(1, 1, 1, half + 1)
        if first:
            v_mask[:, :, -1] = 0     # hide the current row
            h_mask[:, :, :, -1] = 0  # hide the current position
        self.register_buffer("v_mask", v_mask)
        self.register_buffer("h_mask", h_mask)

    def forward(self, v, h, cond=None):
        height, width = v.shape[-2:]
        # Pad then crop, so each output only sees rows above / columns to the left.
        v_pre = F.conv2d(v, self.vertical.weight * self.v_mask, self.vertical.bias,
                         padding=self.vertical.padding)[:, :, :height]
        h_pre = F.conv2d(h, self.horizontal.weight * self.h_mask, self.horizontal.bias,
                         padding=self.horizontal.padding)[:, :, :, :width]
        h_pre = h_pre + self.vert_to_horiz(v_pre)
        if cond is not None and self.cond_v is not None and self.cond_h is not None:
            v_pre = v_pre + self.cond_v(cond)
            h_pre = h_pre + self.cond_h(cond)
        v_out = gate(v_pre)
        h_out = self.dropout(self.horiz_out(gate(h_pre)))
        if not self.first:
            h_out = h_out + h
        return v_out, h_out


class PixelCNN(nn.Module):
    """Gated PixelCNN prior over VQ-VAE code grids.

    Maps a (B, H, W) grid of code indices to (B, num_embeddings, H, W)
    logits, where the logits at each position depend only on the codes
    before it in raster order (left to right, top to bottom).
    """

    def __init__(self, num_embeddings=512, dim=128, num_layers=12, kernel=7, dropout=0.0,
                 cond_embeddings=0):
        super().__init__()
        self.config = {"num_embeddings": num_embeddings, "dim": dim,
                       "num_layers": num_layers, "kernel": kernel, "dropout": dropout,
                       "cond_embeddings": cond_embeddings}
        self.embedding = nn.Embedding(num_embeddings, dim)
        conditional = cond_embeddings > 0
        self.layers = nn.ModuleList([GatedLayer(dim, kernel, first=(i == 0), dropout=dropout,
                                                conditional=conditional)
                                     for i in range(num_layers)])
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_embeddings, 1),
        )
        # Conditioning network: a coarser code grid, upsampled 2x to this grid's size.
        self.cond_net = None
        if conditional:
            self.cond_embedding = nn.Embedding(cond_embeddings, dim)
            self.cond_net = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(inplace=True),
                ResidualBlock(dim, dim // 2), ResidualBlock(dim, dim // 2),
            )

    def condition(self, cond_codes):
        """Features from the conditioning codes at this prior's resolution (no causality limit)."""
        if self.cond_net is None or cond_codes is None:
            return None
        return self.cond_net(self.cond_embedding(cond_codes).permute(0, 3, 1, 2))

    def forward(self, codes, cond_codes=None, cond=None):
        x = self.embedding(codes).permute(0, 3, 1, 2)
        if cond is None:
            cond = self.condition(cond_codes)
        if cond is not None:
            cond = cond[:, :, :codes.size(1)]  # match a row-cropped input during sampling
        v, h = x, x
        for layer in self.layers:
            v, h = layer(v, h, cond)
        return self.output(h)

    @torch.no_grad()
    def sample(self, count, height, width, device, temperature=1.0, cond_codes=None):
        """Sample code grids one position at a time in raster order."""
        codes = torch.zeros(count, height, width, dtype=torch.long, device=device)
        cond = self.condition(cond_codes)
        for row in range(height):
            for col in range(width):
                # Rows below the current one cannot affect it, so they are skipped.
                logits = self(codes[:, :row + 1], cond=cond)[:, :, row, col]
                probs = F.softmax(logits / temperature, dim=1)
                codes[:, row, col] = torch.multinomial(probs, 1).squeeze(1)
        return codes


if __name__ == "__main__":
    prior = PixelCNN()
    print("PixelCNN logits:", tuple(prior(torch.zeros(2, 32, 64, dtype=torch.long)).shape))
    print(f"PixelCNN parameters: {count_parameters(prior):,}")

    vqvae = VQVAE()
    images = torch.rand(2, 1, 128, 256)
    recon, vq_loss, codes = vqvae(images)
    print("VQ-VAE reconstruction:", tuple(recon.shape), "codes:", tuple(codes.shape))
    print(f"VQ-VAE parameters: {count_parameters(vqvae):,}")

    vqvae2 = VQVAE2()
    recon, _, _ = vqvae2(images)
    top, bottom = vqvae2.encode_levels(images)
    print("VQ-VAE-2 reconstruction:", tuple(recon.shape),
          "top codes:", tuple(top.shape), "bottom codes:", tuple(bottom.shape))
    print(f"VQ-VAE-2 parameters: {count_parameters(vqvae2):,}")

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