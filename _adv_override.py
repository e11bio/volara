"""The pattern the `lazy_ops` docstring actually advertises: a SUBCLASS OVERRIDE."""

import json
import pathlib
import tempfile

import numpy as np
from funlib.persistence import Array, prepare_ds

from volara.datasets import Raw

tmp = pathlib.Path(tempfile.mkdtemp())
p = tmp / "s.zarr"
a = prepare_ds(
    p,
    shape=(5, 4, 3),
    dtype=np.uint16,
    voxel_size=(1, 1, 1),
    offset=(0, 0, 0),
    axis_names=["z", "y", "x"],
    units=["nm"] * 3,
    mode="w",
)
data = np.arange(60, dtype=np.uint16).reshape(5, 4, 3)
a[:] = data


class MyRaw(Raw):
    """Exactly what the docstring tells a user to do: override to add a lazy op."""

    def lazy_ops(self, arr: Array) -> None:
        arr.lazy_op(lambda d: d + 1)


out = {}
r = MyRaw(store=p, scale_shift=(0.5, 0.0), voxel_size=(1, 1, 1), offset=(0, 0, 0))
got = np.asarray(r.array("r")[:])
out["override_no_super__dtype"] = str(got.dtype)
out["override_no_super__sum"] = float(got.sum())
out["scale_shift_survived_the_override"] = bool(
    np.allclose(got, (data.astype(np.float32) + 1) * 0.5)
)
out["scale_shift_was_DROPPED"] = bool(np.array_equal(got, (data + 1)))

print(json.dumps(out, indent=2, sort_keys=True))
