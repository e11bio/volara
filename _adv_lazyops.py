"""Adversarial verification probe for the `Raw.lazy_ops` deletion finding."""

import json
import pathlib
import tempfile

import numpy as np
import zarr
from funlib.persistence import open_ds, prepare_ds

import volara
from volara.datasets import Dataset, Raw

print("volara imported from:", volara.__file__)

tmp = pathlib.Path(tempfile.mkdtemp())


def store(shape, name):
    p = tmp / name
    a = prepare_ds(
        p,
        shape=shape,
        dtype=np.uint16,
        voxel_size=(1,) * len(shape),
        offset=(0,) * len(shape),
        axis_names=list("tczyx"[-len(shape) :]),
        units=["nm"] * len(shape),
        mode="w",
    )
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    a[:] = data
    return p, data


out = {}
out["Raw_defines_lazy_ops"] = "lazy_ops" in Raw.__dict__
out["Dataset_defines_lazy_ops"] = "lazy_ops" in Dataset.__dict__
out["Raw.lazy_ops is Dataset.lazy_ops"] = Raw.lazy_ops is Dataset.lazy_ops

# ---------------------------------------------------------------- A: scale_shift via the hook
p3, d3 = store((5, 4, 3), "a.zarr")
r = Raw(store=p3, scale_shift=(0.5, 0.0), voxel_size=(1, 1, 1), offset=(0, 0, 0))
arr = open_ds(p3, mode="r")
r.lazy_ops(arr)
got = np.asarray(arr[:])
out["A_hook_scale_shift"] = {"dtype": str(got.dtype), "sum": float(got.sum())}
out["A_public_array_scale_shift"] = {
    "dtype": str(np.asarray(r.array("r")[:]).dtype),
    "sum": float(np.asarray(r.array("r")[:]).sum()),
}

# ---------------------------------------------------------------- B: flip via the SAME hook
r_flip = Raw(store=p3, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0))
arr_b = open_ds(p3, mode="r")
r_flip.lazy_ops(arr_b)
got_b = np.asarray(arr_b[:])
out["B_hook_flip"] = {
    "dtype": str(got_b.dtype),
    "identical_to_untouched_store": bool(np.array_equal(got_b, d3)),
    "equals_flipped": bool(np.array_equal(got_b, d3[::-1])),
}
out["B_public_array_flip"] = bool(
    np.array_equal(np.asarray(r_flip.array("r")[:]), d3[::-1])
)

# ---------------------------------------------------------------- C: channels via the SAME hook
p4, d4 = store((4, 5, 4, 3), "c.zarr")
r_ch = Raw(store=p4, channels=1, voxel_size=(1,) * 4, offset=(0,) * 4)
arr_c = open_ds(p4, mode="r")
r_ch.lazy_ops(arr_c)
got_c = np.asarray(arr_c[:])
out["C_hook_channels"] = {
    "shape": list(got_c.shape),
    "identical_to_untouched_store": bool(np.array_equal(got_c, d4)),
    "equals_selected": bool(got_c.shape == d4[1].shape and np.array_equal(got_c, d4[1])),
}
out["C_public_array_channels"] = bool(
    np.array_equal(np.asarray(r_ch.array("r")[:]), d4[1])
)

# ------------------------------------------ D: mixed Raw -- was the hook ever a complete apply?
r_mix = Raw(
    store=p4, channels=1, scale_shift=(0.5, 0.0), voxel_size=(1,) * 4, offset=(0,) * 4
)
arr_d = open_ds(p4, mode="r")
r_mix.lazy_ops(arr_d)
got_d = np.asarray(arr_d[:])
pub_d = np.asarray(r_mix.array("r")[:])
out["D_hook_mixed"] = {"shape": list(got_d.shape), "dtype": str(got_d.dtype)}
out["D_public_mixed"] = {"shape": list(pub_d.shape), "dtype": str(pub_d.dtype)}
out["D_hook_equals_public"] = bool(
    got_d.shape == pub_d.shape and np.array_equal(got_d, pub_d)
)

# ---------------------------------------------------------------- E: public path parity, ome_norm
p_om, d_om = store((3, 5, 4), "om.zarr")
meta = tmp / "meta.zarr"
g = zarr.open_group(str(meta), mode="w")
g.attrs["omero"] = {
    "channels": [{"window": {"min": 0.0, "max": 10.0}} for _ in range(3)]
}
r_om = Raw(store=p_om, ome_norm=meta, voxel_size=(1, 1, 1), offset=(0, 0, 0))
pub_om = np.asarray(r_om.array("r")[:])
out["E_public_ome_norm"] = {"dtype": str(pub_om.dtype), "sum": round(float(pub_om.sum()), 4)}
arr_e = open_ds(p_om, mode="r")
r_om.lazy_ops(arr_e)
got_e = np.asarray(arr_e[:])
out["E_hook_ome_norm"] = {"dtype": str(got_e.dtype), "sum": round(float(got_e.sum()), 4)}

# ---------------------------------------------------------------- F: public path parity, stack
other = Raw(store=p_om, voxel_size=(1, 1, 1), offset=(0, 0, 0))
r_st = Raw(store=p_om, stack=other, voxel_size=(1, 1, 1), offset=(0, 0, 0))
pub_st = np.asarray(r_st.array("r")[:])
out["F_public_stack"] = {"shape": list(pub_st.shape), "sum": float(pub_st.sum())}

print(json.dumps(out, indent=2, sort_keys=True))
