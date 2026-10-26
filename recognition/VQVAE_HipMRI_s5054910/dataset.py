# --------------------------------------------------------------------
# File:    dataset.py
# Author:  Gurkirat Singh Randhawa (s5054910)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Loads HipMRI 2D slices: transpose, centre-crop, per-slice
#          normalisation, patient-level splits and a leakage check.
# --------------------------------------------------------------------
"""
dataset.py - Data loading for the HipMRI 2D prostate MRI slices.

Loads the Nifti slices from the HipMRI study (keras_slices_data), brings every
slice to one common shape, normalises intensities to [0, 1] and serves them
through PyTorch DataLoaders.

Data audit findings this loader is built around (run on Rangpur, Oct 2026):
    * Files are named  case_<id>_week_<n>_slice_<k>.nii.gz
      -> "case_<id>" identifies the patient.
    * Provided split:  train 31 patients / 11,460 slices
                       validate 4 patients / 660 slices
                       test 3 patients / 540 slices
      No patient appears in more than one split, so the provided split is
      already patient-level and is used unchanged.
    * On disk, 12,600 slices are 256 x 128 and 60 slices are 256 x 144.
      -> the larger slices are centre-cropped to match the rest.
    * The arrays are stored with the body's left-right axis first, so a raw
      slice displays sideways.
      -> each slice is transposed to 128 x 256 (height x width), the usual
         axial view with the body wider than it is tall.
    * Raw intensities are unnormalised (e.g. 0 to 411 for one slice).

Run this file directly for a quick check:
    python3 dataset.py --max-slices 64 --grid sample_grid.png
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Location of the data on the Rangpur cluster.
DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

# Sub-folder holding the MRI images for each split (the *_seg_* folders hold
# segmentation masks, which image generation does not need).
SPLIT_DIRS = {
    "train": "keras_slices_train",
    "validate": "keras_slices_validate",
    "test": "keras_slices_test",
}

# Every slice is brought to this (height, width), after transposing.
TARGET_SHAPE = (128, 256)

# Intensities above this quantile of a slice are clipped before scaling, so a
# few very bright pixels do not squash the rest of the image towards black.
CLIP_QUANTILE = 0.995

_FILENAME_PATTERN = re.compile(r"^(case_\d+)_week_(\d+)_slice_(\d+)\.nii(\.gz)?$")


def parse_filename(filename):
    """Split a slice file name into (patient_id, week, slice_index).

    Example: "case_004_week_0_slice_10.nii.gz" -> ("case_004", 0, 10)
    """
    match = _FILENAME_PATTERN.match(os.path.basename(filename))
    if match is None:
        raise ValueError(f"Unexpected file name: {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def list_slices(root, split):
    """Return the paths of all slices in a split, in patient/week/slice order."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"split must be one of {list(SPLIT_DIRS)}, got {split!r}")
    folder = os.path.join(root, SPLIT_DIRS[split])
    names = [n for n in os.listdir(folder) if _FILENAME_PATTERN.match(n)]
    # Sort numerically so slice_2 comes before slice_10.
    names.sort(key=parse_filename)
    return [os.path.join(folder, n) for n in names]


def patients_in_split(root, split):
    """Return the set of patient ids that have at least one slice in a split."""
    return {parse_filename(path)[0] for path in list_slices(root, split)}


def check_no_patient_leakage(root):
    """Verify that no patient appears in more than one split.

    Returns a dict of {split: number_of_patients}. Raises ValueError if any
    patient is shared between two splits, because that would let the model be
    evaluated on patients it was trained on.
    """
    patients = {split: patients_in_split(root, split) for split in SPLIT_DIRS}
    splits = list(SPLIT_DIRS)
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            shared = patients[first] & patients[second]
            if shared:
                raise ValueError(
                    f"Patient leakage between {first} and {second}: {sorted(shared)}"
                )
    return {split: len(ids) for split, ids in patients.items()}


def centre_crop(image, target_shape=TARGET_SHAPE):
    """Centre-crop a 2D tensor to target_shape (height, width)."""
    height, width = image.shape
    target_h, target_w = target_shape
    if height < target_h or width < target_w:
        raise ValueError(
            f"Slice of shape {(height, width)} is smaller than target {target_shape}"
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return image[top:top + target_h, left:left + target_w]


def normalise(image, clip_quantile=CLIP_QUANTILE):
    """Scale a slice to the range [0, 1].

    The slice is clipped at its own clip_quantile intensity and divided by
    that value. Normalising per slice removes the scanner-dependent intensity
    scale, which differs between patients and scans.
    """
    upper = torch.quantile(image.flatten(), clip_quantile)
    if upper <= 0:
        # Completely empty slice: return zeros rather than dividing by zero.
        return torch.zeros_like(image)
    return image.clamp(min=0.0, max=upper.item()) / upper


def load_slice(path):
    """Load one Nifti slice as a float tensor of shape (1, 128, 256) in [0, 1]."""
    data = nib.load(path).get_fdata(caching="unchanged", dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]
    if data.ndim == 3:
        # Some HipMRI files carry an extra singleton dimension.
        data = data[:, :, 0]
    # Stored as (left-right, front-back); transpose so the body lies
    # horizontally, as in a standard axial view.
    data = data.T
    image = torch.from_numpy(np.ascontiguousarray(data))
    image = centre_crop(image)
    image = normalise(image)
    return image.unsqueeze(0)  # add the channel dimension


class HipMRISlices(Dataset):
    """HipMRI 2D prostate MRI slices for one split.

    Each item is a float tensor of shape (1, 128, 256) with values in [0, 1].
    There are no labels because the task is unconditional image generation.

    Args:
        root: Folder containing the keras_slices_* sub-folders.
        split: "train", "validate" or "test".
        preload: Load every slice into memory once (about 1.5 GB for the
            training split). Much faster per epoch than re-reading the
            compressed files; set False on a memory-limited machine.
        augment: Apply a random left-right flip. Off by default; only use it
            for training, never for evaluation.
        max_slices: Keep only the first N slices. For quick smoke tests.
    """

    def __init__(self, root=DEFAULT_ROOT, split="train", preload=True,
                 augment=False, max_slices=None):
        self.split = split
        self.augment = augment
        self.paths = list_slices(root, split)
        if max_slices is not None:
            self.paths = self.paths[:max_slices]

        self.images = None
        if preload:
            loaded = [load_slice(p) for p in tqdm(self.paths, desc=f"Loading {split}")]
            self.images = torch.stack(loaded)

    def __len__(self):
        """Number of slices in the split."""
        return len(self.paths)

    def __getitem__(self, index):
        """One normalised (1, 128, 256) slice in [0, 1]."""
        if self.images is not None:
            image = self.images[index]
        else:
            image = load_slice(self.paths[index])
        if self.augment and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[-1])
        return image

    def patient_of(self, index):
        """Return the patient id of the slice at a given index."""
        return parse_filename(self.paths[index])[0]


def get_dataloaders(root=DEFAULT_ROOT, batch_size=64, num_workers=2,
                    preload=True, augment=False, max_slices=None):
    """Build the train, validation and test DataLoaders.

    The leakage check runs first, so training cannot start on a split that
    shares patients. Only the training loader shuffles and (optionally)
    augments; validation and test are deterministic.
    """
    check_no_patient_leakage(root)

    train_set = HipMRISlices(root, "train", preload, augment, max_slices)
    val_set = HipMRISlices(root, "validate", preload, False, max_slices)
    test_set = HipMRISlices(root, "test", preload, False, max_slices)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def main():
    """Quick self-check: leakage, split sizes, tensor shape and value range."""
    parser = argparse.ArgumentParser(description="Check the HipMRI data loader.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--max-slices", type=int, default=64,
                        help="Slices to load per split for this check.")
    parser.add_argument("--grid", default=None,
                        help="Optional path to save a PNG grid of training slices.")
    args = parser.parse_args()

    counts = check_no_patient_leakage(args.root)
    print("No patient leakage. Patients per split:", counts)
    for split in SPLIT_DIRS:
        print(f"{split}: {len(list_slices(args.root, split))} slices")

    dataset = HipMRISlices(args.root, "train", max_slices=args.max_slices)
    batch = torch.stack([dataset[i] for i in range(len(dataset))])
    print("Batch shape:", tuple(batch.shape))
    print(f"Value range: {batch.min().item():.3f} to {batch.max().item():.3f}, "
          f"mean {batch.mean().item():.3f}")

    if args.grid:
        from torchvision.utils import save_image
        save_image(batch[:32], args.grid, nrow=4)
        print("Saved sample grid to", args.grid)


if __name__ == "__main__":
    main()
