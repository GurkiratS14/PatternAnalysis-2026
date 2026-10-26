# --------------------------------------------------------------------
# File:    tests/test_metrics.py
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Checks the hand-written SSIM, FID and KID against reference
#          implementations (scikit-image and SciPy).
# --------------------------------------------------------------------
"""
Metric tests. Every number in the README comes from metrics.py, so each
metric is compared with an independent reference implementation.

Run with:  python3 -m pytest tests/      or      python3 tests/test_metrics.py
"""

import os
import sys

import numpy as np
import torch
from scipy import linalg
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (best_match_ssim, frechet_distance, kernel_inception_distance,  # noqa: E402
                     mean_pairwise_ssim, ssim_matrix, ssim_pairs)

torch.manual_seed(0)
IMAGES = torch.rand(4, 1, 64, 96)


def reference_ssim(first, second):
    """scikit-image SSIM with the same settings: Gaussian window, sigma 1.5, data range 1."""
    score = structural_similarity(first[0].numpy(), second[0].numpy(), data_range=1.0,
                                  gaussian_weights=True, sigma=1.5,
                                  use_sample_covariance=False)
    return float(score)  # type: ignore[arg-type]


def test_ssim_of_identical_images_is_one():
    """SSIM(x, x) = 1 for every image."""
    assert torch.allclose(ssim_pairs(IMAGES, IMAGES), torch.ones(4), atol=1e-5)


def test_ssim_matches_scikit_image():
    """Paired SSIM agrees with scikit-image on noisy versions of the same images."""
    noisy = (IMAGES + 0.1 * torch.randn_like(IMAGES)).clamp(0, 1)
    ours = ssim_pairs(IMAGES, noisy)
    for i in range(len(IMAGES)):
        assert abs(ours[i].item() - reference_ssim(IMAGES[i], noisy[i])) < 1e-3


def test_ssim_matrix_agrees_with_pairs():
    """The all-pairs matrix holds the same values as the one-to-one function."""
    matrix = ssim_matrix(IMAGES, IMAGES)
    for i in range(len(IMAGES)):
        for j in range(len(IMAGES)):
            pair = ssim_pairs(IMAGES[i:i + 1], IMAGES[j:j + 1])[0]
            assert abs(matrix[i, j].item() - pair.item()) < 1e-5


def test_best_match_finds_the_copy():
    """A query that is an exact copy of reference 2 is matched to index 2 with SSIM 1."""
    scores, indices = best_match_ssim(IMAGES[2:3], IMAGES, torch.device("cpu"),
                                      description="test")
    assert indices[0] == 2 and abs(scores[0].item() - 1) < 1e-5


def test_pairwise_ssim_is_one_for_identical_set():
    """A set of identical images has no diversity: mean pairwise SSIM is 1."""
    same = IMAGES[:1].repeat(4, 1, 1, 1)
    assert abs(mean_pairwise_ssim(same, torch.device("cpu")) - 1) < 1e-5


def reference_fid(first, second):
    """FID with SciPy's general matrix square root, as in the original implementation."""
    mu1, mu2 = first.mean(0).numpy(), second.mean(0).numpy()
    s1, s2 = np.cov(first.numpy(), rowvar=False), np.cov(second.numpy(), rowvar=False)
    root = np.real(linalg.sqrtm(s1 @ s2))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(root))


def test_fid_matches_scipy():
    """Our eigenvalue-based FID equals the SciPy reference."""
    first = torch.randn(300, 32, dtype=torch.float64)
    second = torch.randn(300, 32, dtype=torch.float64) * 1.3 + 0.5
    assert abs(frechet_distance(first, second) - reference_fid(first, second)) < 1e-6


def test_fid_of_a_set_with_itself_is_zero():
    """Identical feature sets give FID 0."""
    features = torch.randn(200, 16, dtype=torch.float64)
    assert abs(frechet_distance(features, features)) < 1e-6


def test_kid_separates_same_and_different_distributions():
    """KID is about 0 for samples of one distribution and clearly positive otherwise."""
    first = torch.randn(400, 32, dtype=torch.float64)
    same = torch.randn(400, 32, dtype=torch.float64)
    shifted = torch.randn(400, 32, dtype=torch.float64) + 0.5
    kid_same, std_same = kernel_inception_distance(first, same, subsets=20, subset_size=100)
    kid_shifted, _ = kernel_inception_distance(first, shifted, subsets=20, subset_size=100)
    assert abs(kid_same) < 3 * std_same + 1e-3
    assert kid_shifted > 10 * abs(kid_same)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("passed:", name)
