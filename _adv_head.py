import tempfile, pathlib, numpy as np
import volara, volara.datasets as D
print("MODULE:", volara.__file__)
print("DATASETS:", D.__file__)
from funlib.persistence import prepare_ds

tmp = pathlib.Path(tempfile.mkdtemp())
def store(name, shape=(4,4,4)):
    p = tmp / name
    a = prepare_ds(p, shape=shape, dtype=np.uint16, voxel_size=(1,)*len(shape),
                   offset=(0,)*len(shape), axis_names=list("zyx"[-len(shape):]),
                   units=["nm"]*len(shape), mode="w")
    a[:] = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    return p

# 1. flip -> refused up front
p1 = store("w1.zarr")
try:
    D.Labels(store=p1, flip=[0]).array("a")
    print("1 flip a(): NO REFUSAL")
except ValueError as e:
    print("1 flip a(): ValueError", str(e)[:90])

# 2. ops -> opens, non-writeable, setitem raises
p2 = store("w2.zarr")
arr = D.Labels(store=p2, ops=[lambda d: d + 1]).array("a")
print("2 ops a(): opened OK, is_writeable =", arr.is_writeable)
try:
    arr[np.s_[0:1, 0:1, 0:1]] = 7
    print("2 write: NO ERROR")
except RuntimeError as e:
    print("2 write: RuntimeError", " ".join(str(e).split())[:95])
