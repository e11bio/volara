"""Counter-probe: the SAME override WITHOUT super(), and a Dataset (non-Raw) subclass."""
import tempfile, pathlib
import numpy as np
from funlib.persistence import prepare_ds
import volara.datasets as D
from volara.datasets import Raw, Labels

print("MODULE:", D.__file__)

tmp = pathlib.Path(tempfile.mkdtemp())
def store(name, shape):
    p = tmp / name
    a = prepare_ds(p, shape=shape, dtype=np.uint16, voxel_size=(1,)*len(shape),
                   offset=(0,)*len(shape), axis_names=list("czyx"[-len(shape):]),
                   units=["nm"]*len(shape), mode="w")
    d = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    a[:] = d
    return p, d

# --- A: override WITHOUT super() ---
class NoSuperRaw(Raw):
    def lazy_ops(self, arr):
        arr.lazy_op(lambda d: d.astype(np.float32) + 1000.0)

p, data = store("a.zarr", (2, 2, 2))
out = np.asarray(NoSuperRaw(store=p, scale_shift=(2.0, 0.0), voxel_size=(1,)*3, offset=(0,)*3).array("r")[:])
print("NO-SUPER result[0,0]:", out[0, 0])
print("  scale_shift APPLIED at all? ", not np.array_equal(out, data.astype(np.float32) + 1000.0))
print("  == data+1000 (scale_shift SILENTLY DROPPED):", np.array_equal(out, data.astype(np.float32)+1000.0))
print("  == (data+1000)*2 (scale_shift kept):        ", np.array_equal(out, (data.astype(np.float32)+1000.0)*2))

# --- B: super() called LAST instead of first ---
class SuperLastRaw(Raw):
    def lazy_ops(self, arr):
        arr.lazy_op(lambda d: d.astype(np.float32) + 1000.0)
        super().lazy_ops(arr)

out2 = np.asarray(SuperLastRaw(store=p, scale_shift=(2.0, 0.0), voxel_size=(1,)*3, offset=(0,)*3).array("r")[:])
print("SUPER-LAST result[0,0]:", out2[0, 0])

# --- C: non-Raw Dataset subclass calling super() ---
class MyLabels(Labels):
    def lazy_ops(self, arr):
        super().lazy_ops(arr)
        arr.lazy_op(lambda d: d.astype(np.float32) + 1000.0)

out3 = np.asarray(MyLabels(store=p, voxel_size=(1,)*3, offset=(0,)*3).array("r")[:])
print("LABELS-subclass result[0,0]:", out3[0, 0])
