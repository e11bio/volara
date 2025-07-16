import multiprocessing as mp

mp.set_start_method("fork", force=True)

import os
from pathlib import Path

from funlib.geometry import Coordinate

from volara.blockwise import AffAgglom, ExtractFrags, GraphMWS, Relabel
from volara.datasets import Affs, Labels, Raw
from volara.dbs import SQLite
from volara.lut import LUT

os.chdir(Path(__file__).parent)
print(Path.cwd())

raw = Raw(store="sample_A+_20160601.zarr/raw", scale_shift=(1 / 255, 0), writable=False)  # type: ignore[arg-type]
affs = Affs(store="sample_A+_20160601.zarr/affs", writable=False)  # type: ignore[arg-type]

fragments_graph = SQLite(
    path="sample_A+_20160601.zarr/fragments.db",  # type: ignore[arg-type]
    edge_attrs={"xy_aff": "float", "z_aff": "float", "lr_aff": "float"},
)
fragments_dataset = Labels(store="sample_A+_20160601.zarr/fragments")  # type: ignore[arg-type]
segments_dataset = Labels(store="sample_A+_20160601.zarr/segments")  # type: ignore[arg-type]

block_size = raw.array("r")._source_data.chunks

# Generate fragments in blocks
extract_frags = ExtractFrags(
    db=fragments_graph,
    affs_data=affs,
    frags_data=fragments_dataset,
    block_size=block_size,
    context=Coordinate(6, 12, 12),
    bias=[-0.6] + [-0.4] * 2 + [-0.6] * 2 + [-0.8] * 2,
    strides=(
        [Coordinate(1, 1, 1)] * 3
        + [Coordinate(1, 3, 3)] * 2  # We use larger strides for larger affinities
        + [Coordinate(1, 6, 6)] * 2  # This is to avoid excessive splitting
    ),
    randomized_strides=True,  # converts strides to probabilities of sampling affinities (1/prod(stride))
    remove_debris=64,  # remove excessively small fragments
    num_workers=4,
)

# Generate agglomerated edge scores between fragments via mean affinity accross all edges connecting two fragments
aff_agglom = AffAgglom(
    db=fragments_graph,
    affs_data=affs,
    frags_data=fragments_dataset,
    block_size=block_size,
    context=Coordinate(3, 6, 6) * 1,
    scores={
        "z_aff": affs.neighborhood[0:1],
        "xy_aff": affs.neighborhood[1:3],
        "lr_aff": affs.neighborhood[3:],
    },
    num_workers=4,
)

# Run mutex watershed again, this time on the fragment graph with agglomerated edges
# instead of the voxel graph of affinities
lut = LUT(path="sample_A+_20160601.zarr/lut.npz")  # type: ignore[arg-type]
total_roi = raw.array("r").roi
graph_mws = GraphMWS(
    db=fragments_graph,
    lut=lut,
    weights={"xy_aff": (1, -0.4), "z_aff": (1, -0.6), "lr_aff": (1, -0.6)},
    roi=total_roi,
)

# Relabel the fragments into segments
relabel = Relabel(
    lut=lut,
    frags_data=fragments_dataset,
    seg_data=segments_dataset,
    block_size=block_size,
    num_workers=4,
)

pipeline = extract_frags + aff_agglom + graph_mws + relabel
# pipeline.drop()
pipeline.benchmark(multiprocessing=True)
