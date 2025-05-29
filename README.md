[![tests](https://github.com/e11bio/volara/actions/workflows/tests.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/tests.yaml)
[![ruff](https://github.com/e11bio/volara/actions/workflows/ruff.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/ruff.yaml)
[![mypy](https://github.com/e11bio/volara/actions/workflows/mypy.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/mypy.yaml)
<!-- [![codecov](https://codecov.io/gh/e11bio/volara/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/e11bio/volara) -->

<p align="center">
  <img src="https://github.com/e11bio/volara/blob/main/docs/source/_static/Volara%20Logo-white.svg">
</p>

# Volara
Easy application of common blockwise operations for image processing of arbitrarily large volumetric microscopy.

# Motivation
We have been using [Daisy](https://github.com/funkelab/daisy) for scaling our ML pipelines to process large volumetric data. We found that as pipelines became more complex we were re-writing a lot of useful common functions for different projects. We therefore wanted a unified framework to transparently handle some of this functionality through simple abstractions, while maintaining the efficiency and ease-of-use that Daisy offers. 

Some things we wanted to support:
 * Next gen file formats (e.g zarr & ome-zarr)
 * Lazy operations (e.g thresholding, normalizing, slicing, dtype conversions)
 * Standard image to image pytorch model inference
 * Flexible graph support (e.g both sqlite and postgresql)
 * Multiple compute contexts (e.g serial or parallel, local or cluster, cpu or gpu)
 * Completed block tracking and task resuming
 * Syntactically nice task chaining
 * Plugin system for custom tasks

# Getting started

* Volara is available on PyPi and can be installed with `pip install volara`
* For running inference with pre-trained pytorch models, you can also install [volara-torch](https://github.com/e11bio/volara-torch) with `pip install volara-torch`

# Useful links
- [Blog post](https://e11.bio/blog/volara)
- [API Reference](https://e11bio.github.io/volara/api.html)
- [Basic tutorial](https://e11bio.github.io/volara/tutorial.html)
- [Cremi inference tutorial](https://e11bio.github.io/volara-torch/examples/cremi/cremi.html)
- [Cremi affinity agglomeration tutorial](https://e11bio.github.io/volara/examples/cremi/cremi.html)
- [Building a custom task](https://e11bio.github.io/volara/examples/getting_started/basics.html)
- [Daisy Tutorial](https://funkelab.github.io/daisy/tutorial.html#In-Depth-Tutorial)

# Architecture
![](https://github.com/e11bio/volara/blob/main/docs/source/_static/Diagram-transparent%20bg3.png)
This diagram visualizes the lifetime of a block in volara. On the left we are reading array and/or graph data with optional padding for a specific block. This data is then processed, and written to the output on the right. For every block processed we also mark it done in a separate Zarr. Once each worker completes a block, it will fetch the next. This process continues until the full input dataset has been processed.

# Available blockwise operations:
- `ExtractFrags`: Fragment extraction via mutex watershed
- `AffAgglom`: Supervoxel affinity score edge creation
- `GraphMWS`: Global creation of look up tables for fragment -> segment agglomeration
- `Relabel`: Remapping and saving fragments as segments
- `SeededExtractFrags`: Constrained fragment extraction via mutex watershed that accepts skeletonized seed points
- `ArgMax`: Argmax accross predicted probabilities
- `DistanceAgglom`: Supervoxel distance score edge creation, computed between stored supervoxel embeddings. 
- `ComputeShift`: Compute shift between moving and fixed image using phase cross correlation
- `ApplyShift`: Apply computed shift to register moving image to fixed image
- `Threshold`: Intensity threshold an array

# Example pipeline

Below is a simple example pipeline showing how to compute a segmentation from affinities.

```py
from funlib.geometry import Coordinate
from funlib.persistence import open_ds
from pathlib import Path
from volara.blockwise import ExtractFrags, AffAgglom, GraphMWS, Relabel
from volara.datasets import Affs, Labels
from volara.dbs import SQLite
from volara.lut import LUT

file = Path("test.zarr")

block_size = Coordinate(15, 40, 40) * 3
context = Coordinate(15, 40, 40)
bias = [-0.4, -0.7]

affs = Affs(
    store=file / "affinities",
    neighborhood=[
        Coordinate(1, 0, 0),
        Coordinate(0, 1, 0),
        Coordinate(0, 0, 1),
        Coordinate(4, 0, 0),
        Coordinate(0, 8, 0),
        Coordinate(0, 0, 8),
        Coordinate(8, 0, 0),
        Coordinate(0, 16, 0),
        Coordinate(0, 0, 16),
    ],
)

db = SQLite(
    path=file / "db.sqlite",
    edge_attrs={
        "adj_weight": "float",
        "lr_weight": "float",
    },
)

fragments = Labels(store=file / "fragments")

extract_frags = ExtractFrags(
    db=db,
    affs_data=affs,
    frags_data=fragments,
    block_size=block_size,
    context=context,
    bias=[bias[0]] * 3 + [bias[1]] * 6,
    num_workers=10,
)

aff_agglom = AffAgglom(
    db=db,
    affs_data=affs,
    frags_data=fragments,
    block_size=block_size,
    context=context,
    scores={"adj_weight": affs.neighborhood[0:3], "lr_weight": affs.neighborhood[3:]},
    num_workers=10,
)

lut = LUT(path=file / "lut.npz")
roi = open_ds(file / "affinities").roi

global_mws = GraphMWS(
    db=db,
    lut=lut,
    weights={"adj_weight": (1.0, bias[0]), "lr_weight": (1.0, bias[1])},
    roi=[roi.get_begin(), roi.get_shape()],
)

relabel = Relabel(
    frags_data=fragments,
    seg_data=Labels(store=file / "segments"),
    lut=lut,
    block_size=block_size,
    num_workers=5,
)

pipeline = extract_frags + aff_agglom + global_mws + relabel

pipeline.run_blockwise()
```
