# --------------------------------------------------------------------
# File:    tests/test_pixelcnn_causality.py
# Author:  Gurkirat Singh Randhawa (50549102)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Proves the PixelCNN masks are correct: the prediction at each
#          grid position never depends on that code or any later code.
# --------------------------------------------------------------------
"""Checks the PixelCNN masks: changing one code may only move the logits of later raster positions.
Run: python3 -m pytest tests/
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import PixelCNN  # noqa: E402

HEIGHT, WIDTH = 8, 10  # a small grid keeps the test fast


def changed_positions(prior, row, col, cond_codes=None):
    """Grid positions whose logits change when only the code at (row, col) changes."""
    torch.manual_seed(0)
    codes = torch.randint(0, 512, (1, HEIGHT, WIDTH))
    edited = codes.clone()
    edited[0, row, col] = (codes[0, row, col] + 1) % 512
    with torch.no_grad():
        before = prior(codes, cond_codes=cond_codes)
        after = prior(edited, cond_codes=cond_codes)
    # A position "changed" if any of its 512 logits moved by more than rounding error.
    return (before - after).abs().amax(dim=1)[0] > 1e-5


def assert_only_later_positions_change(changed, row, col):
    """Every changed position must come strictly after (row, col) in raster order."""
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if (r, c) <= (row, col):
                assert not changed[r, c], f"logits at {(r, c)} depend on code at {(row, col)}"


def test_unconditional_prior_is_causal():
    """Checked for a corner, an edge and an interior position, with a deep network."""
    prior = PixelCNN(dim=16, num_layers=6).eval()
    for row, col in [(0, 0), (3, 0), (4, 5), (7, 9)]:
        changed = changed_positions(prior, row, col)
        assert_only_later_positions_change(changed, row, col)


def test_conditional_prior_is_causal():
    """Conditioning on top-level codes must not open a path to later bottom codes."""
    prior = PixelCNN(dim=16, num_layers=6, cond_embeddings=512).eval()
    top = torch.randint(0, 512, (1, HEIGHT // 2, WIDTH // 2))
    for row, col in [(0, 0), (4, 5)]:
        changed = changed_positions(prior, row, col, cond_codes=top)
        assert_only_later_positions_change(changed, row, col)


def test_later_positions_do_see_the_change():
    """Sanity check the other way: the next position does use the code before it."""
    prior = PixelCNN(dim=16, num_layers=6).eval()
    changed = changed_positions(prior, 4, 5)
    assert changed[4, 6], "the position to the right should depend on (4, 5)"
    assert changed[5, 5], "the position below should depend on (4, 5)"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("passed:", name)
