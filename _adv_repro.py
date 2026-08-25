import sys, numpy as np, zarr
from pathlib import Path
import tempfile
from funlib.persistence import prepare_ds
from volara.datasets import Raw

tmp = Path(tempfile.mkdtemp())
p = tmp / "t.zarr"
a = prepare_ds(p, shape=(2, 4, 3, 3), dtype=np.uint16, voxel_size=(1,)*4, offset=(0,)*4,
               axis_names=["c","z","y","x"], units=["nm"]*4, mode="w")
a[:] = np.arange(2*4*3*3, dtype=np.uint16).reshape(2,4,3,3)

# omero metadata that lists THREE channels while the store carries TWO
meta = tmp / "meta.zarr"
g = zarr.open_group(str(meta), mode="w")
g.attrs["omero"] = {"channels": [
    {"window": {"min": 0.0, "max": 100.0}},
    {"window": {"min": 0.0, "max": 200.0}},
    {"window": {"min": 0.0, "max": 300.0}},
]}

r = Raw(store=p, ome_norm=meta, voxel_size=(1,)*4, offset=(0,)*4)
print("Raw.bounds (old computation, still used by attrs):", r.bounds)
try:
    out = np.asarray(r.array("r")[:])
    print("read OK, shape", out.shape, "max", out.max())
except Exception as e:
    print("READ FAILED:", type(e).__name__, e)
