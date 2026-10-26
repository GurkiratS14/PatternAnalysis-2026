# --------------------------------------------------------------------
# File:    tests/test_modules.py
# Author:  Gurkirat Singh Randhawa (50549102)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Checks that every model in modules.py produces the expected
#          output shapes and has the parameter counts reported in the README.
# --------------------------------------------------------------------
"""Output-shape and parameter-count tests for every model; runs on the CPU in a few seconds.
Run: python3 -m pytest tests/
"""

import os
import sys

import torch

# Make the project folder importable when this file is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import (VQVAE, VQVAE2, Critic, Discriminator, Generator, PixelCNN,  # noqa: E402
                     count_parameters)

IMAGES = torch.rand(2, 1, 128, 256)  # two slices in [0, 1], the dataset's format


def test_generator_output():
    """Noise -> one 128 x 256 image per vector, inside tanh's range [-1, 1]."""
    fake = Generator()(torch.randn(3, 100))
    assert fake.shape == (3, 1, 128, 256)
    assert fake.min() >= -1 and fake.max() <= 1


def test_discriminator_and_critic_give_one_score_per_image():
    """Both return one unbounded score per image."""
    assert Discriminator()(IMAGES).shape == (2,)
    assert Critic()(IMAGES).shape == (2,)


def test_vqvae_shapes():
    """Reconstruction matches the input; codes form a 32 x 64 grid of valid indices."""
    model = VQVAE()
    recon, vq_loss, codes = model(IMAGES)
    assert recon.shape == IMAGES.shape
    assert vq_loss.ndim == 0
    assert codes.shape == (2, 32, 64)
    assert codes.min() >= 0 and codes.max() < 512


def test_vqvae2_levels():
    """A 16 x 32 top grid and a 32 x 64 bottom grid, which decode back to images."""
    model = VQVAE2().eval()
    top, bottom = model.encode_levels(IMAGES)
    assert top.shape == (2, 16, 32)
    assert bottom.shape == (2, 32, 64)
    assert model.decode_levels([top, bottom]).shape == IMAGES.shape


def test_pixelcnn_logits_shape():
    """One 512-way distribution per grid position, with and without conditioning."""
    codes = torch.randint(0, 512, (2, 32, 64))
    assert PixelCNN(num_layers=2)(codes).shape == (2, 512, 32, 64)
    conditional = PixelCNN(num_layers=2, cond_embeddings=512)
    top = torch.randint(0, 512, (2, 16, 32))
    assert conditional(codes, cond_codes=top).shape == (2, 512, 32, 64)


def test_parameter_counts_match_readme():
    """The parameter counts quoted in the README's resource table."""
    assert count_parameters(VQVAE()) == 690_753
    assert count_parameters(VQVAE2()) == 1_653_889
    assert count_parameters(PixelCNN()) == 13_724_800
    assert count_parameters(PixelCNN(cond_embeddings=512)) == 15_009_024
    assert count_parameters(Generator()) == 14_422_912
    assert count_parameters(Discriminator()) == 11_178_752


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("passed:", name)
