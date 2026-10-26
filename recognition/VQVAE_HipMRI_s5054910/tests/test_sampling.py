# --------------------------------------------------------------------
# File:    tests/test_sampling.py
# Author:  Gurkirat Singh Randhawa (50549102)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Checks the top-k / top-p filters used when sampling the PixelCNN
#          priors, and that the default sampling is unchanged.
# --------------------------------------------------------------------
"""Tests for filter_logits and PixelCNN.sample with top-k / top-p."""
import torch

from modules import PixelCNN, filter_logits


def test_defaults_leave_logits_unchanged():
    logits = torch.randn(4, 512)
    assert torch.equal(filter_logits(logits), logits)


def test_top_k_keeps_exactly_k_codes():
    logits = torch.randn(4, 512)
    kept = torch.isfinite(filter_logits(logits, top_k=16)).sum(dim=1)
    assert (kept == 16).all()


def test_top_p_keeps_smallest_set_reaching_p():
    # Probabilities 0.5, 0.3, 0.15, 0.05: p = 0.8 needs the first two codes only.
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()
    kept = torch.isfinite(filter_logits(logits, top_p=0.8))
    assert kept.tolist() == [[True, True, False, False]]


def test_top_p_always_keeps_most_likely_code():
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    assert torch.isfinite(filter_logits(logits, top_p=0.01)).sum() == 1


def test_default_sampling_is_unchanged():
    # With default filters, sample() must draw exactly what it drew before the option existed.
    prior = PixelCNN(num_embeddings=8, dim=16, num_layers=2).eval()
    torch.manual_seed(0)
    a = prior.sample(2, 3, 4, "cpu")
    torch.manual_seed(0)
    b = prior.sample(2, 3, 4, "cpu", top_k=0, top_p=1.0)
    assert torch.equal(a, b)


def test_top_k_one_is_greedy():
    prior = PixelCNN(num_embeddings=8, dim=16, num_layers=2).eval()
    torch.manual_seed(1)
    a = prior.sample(2, 3, 4, "cpu", top_k=1)
    torch.manual_seed(2)
    b = prior.sample(2, 3, 4, "cpu", top_k=1)
    assert torch.equal(a, b)  # no randomness left
