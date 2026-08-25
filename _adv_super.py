"""The other half of the title: a subclass that CALLS `super().lazy_ops(arr)`."""

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


class SuperRaw(Raw):
    def lazy_ops(self, arr: Array) -> None:
        super().lazy_ops(arr)
        arr.lazy_op(lambda d: d + 1)


r = SuperRaw(store=p, scale_shift=(0.5, 0.0), voxel_size=(1, 1, 1), offset=(0, 0, 0))
got = np.asarray(r.array("r")[:])

f = data.astype(np.float32)
out = {
    "array_dtype": str(got.dtype),
    "array_sum": float(got.sum()),
    "array_is_UNTOUCHED_raw_uint16": bool(
        got.dtype == np.uint16 and np.array_equal(got, data)
    ),
    "scale_shift_applied_at_all": bool(got.dtype == np.float32),
    "order_scale_shift_THEN_plus1": bool(np.allclose(got, f * 0.5 + 1)),
    "order_plus1_THEN_scale_shift": bool(np.allclose(got, (f + 1) * 0.5)),
}
print(json.dumps(out, indent=2, sort_keys=True))
