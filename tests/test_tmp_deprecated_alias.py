"""The old ``volara.tmp`` path must keep working, but warn.

``volara.tmp`` was renamed to ``volara.segment_utils``. The old name is baked into
the published tutorial, so the alias has to stay importable for now.
"""

import importlib
import sys

import numpy as np
import pytest

import volara.segment_utils

PUBLIC_NAMES = [
    "filter_mapping_to_block",
    "prepare_mapping",
    "replace_values",
    "replace_values_sorted",
    "seg_to_affgraph",
    "warmup_replace_values_sorted",
]


def _fresh_import():
    """Import volara.tmp with a cold module cache so the warning fires."""
    sys.modules.pop("volara.tmp", None)
    return importlib.import_module("volara.tmp")


def test_importing_old_path_warns():
    with pytest.warns(DeprecationWarning, match="renamed to volara.segment_utils"):
        _fresh_import()


def test_old_path_reexports_are_the_same_objects():
    with pytest.warns(DeprecationWarning):
        tmp = _fresh_import()

    for name in PUBLIC_NAMES:
        assert getattr(tmp, name) is getattr(volara.segment_utils, name), name


def test_old_path_still_computes():
    """The aliased kernels are usable, not just importable."""
    with pytest.warns(DeprecationWarning):
        tmp = _fresh_import()

    seg = np.array([[1, 1, 2, 2]] * 4, dtype=np.uint64)
    affs = tmp.seg_to_affgraph(seg, nhood=[[1, 0], [0, 1]])
    assert affs.shape == (2, 4, 4)
    assert np.all(affs[1, :, 1] == 0)

    arr = np.array([1, 2, 3, 99], dtype=np.int64)
    result = tmp.replace_values(
        arr, np.array([1, 2], dtype=np.int64), np.array([10, 20], dtype=np.int64)
    )
    np.testing.assert_array_equal(result, [10, 20, 3, 99])
