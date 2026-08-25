import sys, tempfile, pathlib
import numpy as np
from funlib.persistence import prepare_ds
import volara, volara.datasets as D
from volara.datasets import Raw

print("MODULE:", D.__file__)
print("Raw has OWN lazy_ops:", "lazy_ops" in Raw.__dict__)
print("Raw has OWN resolved_ops:", "resolved_ops" in Raw.__dict__)

class MyRaw(Raw):
    def lazy_ops(self, arr):
        super().lazy_ops(arr)                       # base: applies scale_shift here
        arr.lazy_op(lambda d: d.astype(np.float32) + 1000.0)

tmp = pathlib.Path(tempfile.mkdtemp())
p = tmp / "t.zarr"
shape = (2, 2, 2)
a = prepare_ds(p, shape=shape, dtype=np.uint16, voxel_size=(1,)*3, offset=(0,)*3,
               axis_names=list("zyx"), units=["nm"]*3, mode="w")
data = np.arange(8, dtype=np.uint16).reshape(shape)
a[:] = data

r = MyRaw(store=p, scale_shift=(2.0, 0.0), voxel_size=(1,)*3, offset=(0,)*3)
out = np.asarray(r.array("r")[:])
print("result[0,0]:", out[0, 0])
print("matches (data*2)+1000 :", np.array_equal(out, data.astype(np.float32)*2 + 1000.0))
print("matches (data+1000)*2 :", np.array_equal(out, (data.astype(np.float32)+1000.0)*2))
