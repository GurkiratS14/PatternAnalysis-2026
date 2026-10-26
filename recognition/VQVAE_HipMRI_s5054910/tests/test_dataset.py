# --------------------------------------------------------------------
# File:    tests/test_dataset.py
# Author:  Gurkirat Singh Randhawa (50549102)
# Course:  COMP3710 Pattern Recognition and Analysis, The University of Queensland
# Date:    October 2026
# Purpose: Checks the slice pre-processing (crop, normalisation, file names)
#          and that the patient-leakage check catches a shared patient.
# --------------------------------------------------------------------
"""Pre-processing and split tests on a tiny fake dataset, so no real HipMRI data is needed.
Run: python3 -m pytest tests/
"""

import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (SPLIT_DIRS, HipMRISlices, centre_crop,  # noqa: E402
                     check_no_patient_leakage, load_slice, normalise, parse_filename)


def make_fake_root(cases_per_split, shape=(256, 128)):
    """Temporary HipMRI-style folder: {split: [case numbers]} -> path of the root."""
    root = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    for split, cases in cases_per_split.items():
        folder = os.path.join(root, SPLIT_DIRS[split])
        os.makedirs(folder)
        for case in cases:
            for index in range(2):
                data = (rng.random(shape) * 400).astype(np.float32)
                name = f"case_{case:03d}_week_0_slice_{index}.nii.gz"
                nib.save(nib.Nifti1Image(data, np.eye(4)), os.path.join(folder, name))
    return root


def test_parse_filename():
    """Patient id, week and slice index are read from the file name."""
    assert parse_filename("case_004_week_0_slice_10.nii.gz") == ("case_004", 0, 10)


def test_centre_crop_removes_equal_rows_top_and_bottom():
    """A 144-row slice loses 8 rows at the top and 8 at the bottom."""
    image = torch.arange(144).float().view(144, 1).repeat(1, 256)
    cropped = centre_crop(image)
    assert cropped.shape == (128, 256)
    assert cropped[0, 0] == 8 and cropped[-1, 0] == 135


def test_normalise_maps_to_unit_range():
    """Values end up in [0, 1], with the 99.5th percentile mapped to 1."""
    image = torch.rand(128, 256) * 400
    image[0, 0] = 10_000  # one very bright outlier pixel
    scaled = normalise(image)
    assert scaled.min() >= 0 and scaled.max() <= 1
    # The outlier is clipped, so ordinary tissue keeps a usable range.
    assert scaled.median() > 0.4


def test_empty_slice_normalises_to_zeros():
    """An all-zero slice must not cause a division by zero."""
    assert torch.equal(normalise(torch.zeros(128, 256)), torch.zeros(128, 256))


def test_load_slice_transposes_and_crops():
    """A 256 x 144 file becomes a (1, 128, 256) tensor in [0, 1]."""
    root = make_fake_root({"train": [1]}, shape=(256, 144))
    folder = os.path.join(root, SPLIT_DIRS["train"])
    image = load_slice(os.path.join(folder, sorted(os.listdir(folder))[0]))
    assert image.shape == (1, 128, 256)
    assert image.min() >= 0 and image.max() <= 1


def test_leakage_check_passes_on_separate_patients():
    """Separate patients in each split: the check returns patient counts."""
    root = make_fake_root({"train": [1, 2], "validate": [3], "test": [4]})
    assert check_no_patient_leakage(root) == {"train": 2, "validate": 1, "test": 1}


def test_leakage_check_catches_shared_patient():
    """The same patient in train and test must stop everything."""
    root = make_fake_root({"train": [1, 2], "validate": [3], "test": [2]})
    try:
        check_no_patient_leakage(root)
    except ValueError as error:
        assert "case_002" in str(error)
    else:
        raise AssertionError("a patient shared between train and test was not detected")


def test_augmentation_only_flips_left_right():
    """A flipped slice is an exact mirror image: no other change to the data."""
    root = make_fake_root({"train": [1]})
    plain = HipMRISlices(root, "train")
    augmented = HipMRISlices(root, "train", augment=True)
    torch.manual_seed(0)
    for index in range(len(plain)):
        image = augmented[index]
        assert torch.equal(image, plain[index]) or torch.equal(image, plain[index].flip(-1))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("passed:", name)
