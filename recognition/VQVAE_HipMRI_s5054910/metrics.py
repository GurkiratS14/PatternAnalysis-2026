"""
metrics.py - Image similarity metrics for evaluating generated slices.

Implements the Structural Similarity Index (SSIM; Wang et al., 2004) in
PyTorch so that many image pairs can be scored at once on the GPU. Uses the
standard settings: 11 x 11 Gaussian window with sigma 1.5, K1 = 0.01,
K2 = 0.03, and images in [0, 1].
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