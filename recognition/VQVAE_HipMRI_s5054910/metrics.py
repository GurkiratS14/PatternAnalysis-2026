# --------------------------------------------------------------------
# File:    metrics.py
# Author:  Gurkirat Singh Randhawa (50549102)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Batched GPU SSIM (paired, best-match, pairwise) and the FID / KID
#          generation-quality metrics.
# --------------------------------------------------------------------
"""Batched GPU SSIM (Wang et al., 2004), FID (Heusel et al., 2017) and KID (Binkowski et al., 2018).
SSIM uses an 11 x 11 Gaussian window (sigma 1.5) on images in [0, 1]; FID and KID use the
pytorch-fid Inception-v3 only to measure images, never to train.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

C1 = 0.01 ** 2
C2 = 0.03 ** 2


def gaussian_kernel(size=11, sigma=1.5):
    """1D Gaussian weights that sum to 1."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def blur(images, kernel):
    """Gaussian-weighted local mean of (N, 1, H, W) images, without padding."""
    kernel = kernel.to(images.device)
    images = F.conv2d(images, kernel.view(1, 1, -1, 1))
    return F.conv2d(images, kernel.view(1, 1, 1, -1))


def ssim_matrix(queries, references, kernel=None):
    """SSIM between every query and every reference image.

    Args:
        queries: (Q, 1, H, W) tensor in [0, 1].
        references: (R, 1, H, W) tensor in [0, 1].
    Returns:
        (Q, R) tensor of mean SSIM values.
    """
    kernel = gaussian_kernel() if kernel is None else kernel
    num_q, num_r = queries.size(0), references.size(0)

    mu_q = blur(queries, kernel)                                    # (Q, 1, h, w)
    mu_r = blur(references, kernel).squeeze(1).unsqueeze(0)         # (1, R, h, w)
    var_q = blur(queries * queries, kernel) - mu_q ** 2
    var_r = (blur(references * references, kernel).squeeze(1).unsqueeze(0)
             - mu_r ** 2)

    # The covariance needs the local mean of q * r for every pair.
    products = queries * references.squeeze(1).unsqueeze(0)         # (Q, R, H, W)
    cross = blur(products.view(num_q * num_r, 1, *queries.shape[-2:]), kernel)
    covariance = cross.view(num_q, num_r, *mu_q.shape[-2:]) - mu_q * mu_r

    ssim_map = (((2 * mu_q * mu_r + C1) * (2 * covariance + C2))
                / ((mu_q ** 2 + mu_r ** 2 + C1) * (var_q + var_r + C2)))
    return ssim_map.mean(dim=(-2, -1))


def ssim_pairs(first, second, kernel=None):
    """SSIM between first[i] and second[i] for each i: (N, 1, H, W) -> (N,)."""
    kernel = gaussian_kernel() if kernel is None else kernel
    mu_a, mu_b = blur(first, kernel), blur(second, kernel)
    var_a = blur(first * first, kernel) - mu_a ** 2
    var_b = blur(second * second, kernel) - mu_b ** 2
    covariance = blur(first * second, kernel) - mu_a * mu_b
    ssim_map = (((2 * mu_a * mu_b + C1) * (2 * covariance + C2))
                / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)))
    return ssim_map.mean(dim=(-3, -2, -1))


@torch.no_grad()
def mean_pairwise_ssim(images, device, query_batch=16):
    """Mean SSIM over all pairs of different images in a set (a diversity measure).

    Lower means a more varied set; a model suffering mode collapse scores high.
    """
    kernel = gaussian_kernel()
    images = images.to(device)
    count = images.size(0)
    total = 0.0
    for start in range(0, count, query_batch):
        total += ssim_matrix(images[start:start + query_batch], images, kernel).sum().item()
    total -= count  # remove each image compared with itself (SSIM 1)
    return total / (count * (count - 1))


@torch.no_grad()
def best_match_ssim(queries, references, device, query_batch=16,
                    reference_batch=256, description="SSIM"):
    """For each query, find the most similar reference image.

    Returns:
        scores: (Q,) highest SSIM of each query against any reference.
        indices: (Q,) index of the reference that gave that score.
    """
    kernel = gaussian_kernel()
    queries = queries.to(device)
    scores = torch.full((queries.size(0),), -1.0, device=device)
    indices = torch.zeros(queries.size(0), dtype=torch.long, device=device)

    for r_start in tqdm(range(0, references.size(0), reference_batch), desc=description):
        refs = references[r_start:r_start + reference_batch].to(device)
        for q_start in range(0, queries.size(0), query_batch):
            q_end = q_start + query_batch
            matrix = ssim_matrix(queries[q_start:q_end], refs, kernel)
            chunk_scores, chunk_indices = matrix.max(dim=1)
            better = chunk_scores > scores[q_start:q_end]
            scores[q_start:q_end][better] = chunk_scores[better]
            indices[q_start:q_end][better] = chunk_indices[better] + r_start

    return scores.cpu(), indices.cpu()


# ----------------------------------------------------------------------------
# FID and KID (generation quality)
# ----------------------------------------------------------------------------

@torch.no_grad()
def inception_features(images, device, batch_size=50):
    """2048-d Inception-v3 pool features of (N, 1, H, W) images in [0, 1]."""
    # Imported here so the SSIM functions work without pytorch-fid installed.
    from pytorch_fid.inception import InceptionV3

    # resize_input scales to 299 x 299 and normalize_input maps [0, 1] to [-1, 1],
    # exactly as in the reference FID implementation.
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]],
                        resize_input=True, normalize_input=True).to(device).eval()
    features = []
    for start in range(0, images.size(0), batch_size):
        # Grayscale MRI is repeated over three channels to match the RGB network.
        batch = images[start:start + batch_size].to(device).float().repeat(1, 3, 1, 1)
        features.append(model(batch)[0].flatten(1).cpu())
    return torch.cat(features).double()


def frechet_distance(first, second):
    """FID between two feature sets: distance between Gaussians fitted to each.

    FID = |mu1 - mu2|^2 + Tr(S1 + S2 - 2 (S1 S2)^1/2). The trace of (S1 S2)^1/2
    equals that of (S1^1/2 S2 S1^1/2)^1/2, a symmetric matrix, so it can be
    computed stably from eigenvalues instead of a general matrix square root.
    """
    mu1, mu2 = first.mean(0), second.mean(0)
    s1, s2 = torch.cov(first.T), torch.cov(second.T)
    # Symmetric square root of S1 from its eigendecomposition (tiny negatives clipped).
    vals, vecs = torch.linalg.eigh(s1)
    s1_half = vecs @ torch.diag(vals.clamp(min=0).sqrt()) @ vecs.T
    inner = torch.linalg.eigvalsh(s1_half @ s2 @ s1_half).clamp(min=0)
    trace_term = s1.trace() + s2.trace() - 2 * inner.sqrt().sum()
    return ((mu1 - mu2).pow(2).sum() + trace_term).item()


def kernel_inception_distance(first, second, subsets=100, subset_size=500, seed=0):
    """KID as (mean, std) over random subsets: unbiased MMD^2 with a cubic kernel.

    Unlike FID, KID has no bias from small sample sizes, so it is the fairer
    comparison when only a few hundred real slices are available.
    """
    generator = torch.Generator().manual_seed(seed)
    size = min(subset_size, len(first), len(second))
    dim = first.size(1)
    scores = []
    for _ in range(subsets):
        x = first[torch.randperm(len(first), generator=generator)[:size]]
        y = second[torch.randperm(len(second), generator=generator)[:size]]
        # Polynomial kernel k(a, b) = (a.b / d + 1)^3, the standard KID choice.
        k_xx = (x @ x.T / dim + 1) ** 3
        k_yy = (y @ y.T / dim + 1) ** 3
        k_xy = (x @ y.T / dim + 1) ** 3
        # Diagonal terms are left out of the within-set means, which makes MMD^2 unbiased.
        off_diag = size * (size - 1)
        mmd = ((k_xx.sum() - k_xx.diagonal().sum()) / off_diag
               + (k_yy.sum() - k_yy.diagonal().sum()) / off_diag
               - 2 * k_xy.mean())
        scores.append(mmd.item())
    scores = torch.tensor(scores)
    return scores.mean().item(), scores.std().item()
