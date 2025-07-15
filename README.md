[![tests](https://github.com/e11bio/volara/actions/workflows/tests.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/tests.yaml)
[![ruff](https://github.com/e11bio/volara/actions/workflows/ruff.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/ruff.yaml)
[![mypy](https://github.com/e11bio/volara/actions/workflows/mypy.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/mypy.yaml)
[![docs](https://github.com/e11bio/volara/actions/workflows/docs.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/docs.yaml)
<!-- [![codecov](https://codecov.io/gh/e11bio/volara/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/e11bio/volara) -->

<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/e11bio/volara/refs/heads/main/docs/source/_static/Volara%20Logo.svg" media="(prefers-color-scheme: light)">
    <source srcset="https://raw.githubusercontent.com/e11bio/volara/refs/heads/main/docs/source/_static/Volara%20Logo-white.svg" media="(prefers-color-scheme: dark)">
    <img src="http://raw.githubusercontent.com/e11bio/volara/refs/heads/main/docs/source/_static/Volara_Logo-white_with_bcg.svg" alt="Volara Logo">
  </picture>
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
![](https://github.com/e11bio/volara/blob/main/docs/source/_static/Diagram-dark%20bg3.png)
This diagram visualizes the lifetime of a block in volara. On the left we are reading array and/or graph data with optional padding for a specific block. This data is then processed, and written to the output on the right. For every block processed we also mark it done in a separate Zarr. Once each worker completes a block, it will fetch the next. This process continues until the full input dataset has been processed.

# Available blockwise operations:
- `ExtractFrags`: Fragment extraction via mutex watershed (using [mwatershed](https://github.com/pattonw/mwatershed))
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

output:

```
Task add
fragments-extract-frags ✔: 100%|█| 75/75 [00:26<00:00,  2.82blocks/s, ⧗=0, ▶=0, ✔=75, ✗=0, ∅=
db-aff-agglom ✔: 100%|█████████| 75/75 [00:35<00:00,  2.14blocks/s, ⧗=0, ▶=0, ✔=75, ✗=0, ∅=0]
lut-graph-mws ✔: 100%|████████████| 1/1 [00:00<00:00,  9.18blocks/s, ⧗=0, ▶=0, ✔=1, ✗=0, ∅=0]
segments-relabel ✔: 100%|██████| 75/75 [00:02<00:00, 32.66blocks/s, ⧗=0, ▶=0, ✔=75, ✗=0, ∅=0]

Execution Summary
-----------------

  Task fragments-extract-frags:

    num blocks : 75
    completed ✔: 75 (skipped 0)
    failed    ✗: 0
    orphaned  ∅: 0

    all blocks processed successfully

  Task db-aff-agglom:

    num blocks : 75
    completed ✔: 75 (skipped 0)
    failed    ✗: 0
    orphaned  ∅: 0

    all blocks processed successfully

  Task lut-graph-mws:

    num blocks : 1
    completed ✔: 1 (skipped 0)
    failed    ✗: 0
    orphaned  ∅: 0

    all blocks processed successfully

  Task segments-relabel:

    num blocks : 75
    completed ✔: 75 (skipped 0)
    failed    ✗: 0
    orphaned  ∅: 0

    all blocks processed successfully
```

# Example custom task

Simple example argmax task. See [here](https://e11bio.github.io/volara/examples/getting_started/basics.html) for more info

```py
from contextlib import contextmanager
from daisy import Block
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds
from volara.blockwise import BlockwiseTask
import logging
import numpy as np
import shutil
import zarr


logging.basicConfig(level=logging.INFO)


class Argmax(BlockwiseTask):
    task_type: str = "argmax"
    fit: str = "shrink"
    read_write_conflict: bool = False

    @property
    def task_name(self) -> str:
        return "simple-argmax"

    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0, 0), (10, 10, 10))

    @property
    def write_size(self) -> Coordinate:
        return Coordinate((5, 5, 5))

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0, 0, 0))

    def init(self):
        in_array = prepare_ds(
            f"{self.task_name}/data.zarr/in_array",
            shape=(3, 10, 10, 10),
            chunk_shape=(3, 5, 5, 5),
            offset=(0, 0, 0),
        )
        np.random.seed(0)
        in_array[:] = np.random.randint(0, 10, size=in_array.shape)

        prepare_ds(
            f"{self.task_name}/data.zarr/out_array",
            shape=(10, 10, 10),
            chunk_shape=(5, 5, 5),
            offset=(0, 0, 0),
        )

    def drop_artifacts(self):
        shutil.rmtree(f"{self.task_name}/data.zarr/in_array")
        shutil.rmtree(f"{self.task_name}/data.zarr/out_array")

    @contextmanager
    def process_block_func(self):
        in_array = open_ds(
            f"{self.task_name}/data.zarr/in_array",
            mode="r+",
        )
        out_array = open_ds(
            f"{self.task_name}/data.zarr/out_array",
            mode="r+",
        )

        def process_block(block: Block) -> None:
            in_data = in_array[block.read_roi]
            out_data = in_data.argmax(axis=0)
            out_array[block.write_roi] = out_data

        yield process_block


if __name__ == "__main__":

    argmax_task = Argmax()
    argmax_task.run_blockwise(multiprocessing=False)

    print(zarr.open(f"{argmax_task.task_name}/data.zarr/in_array")[:, :, 0, 0])
    print(zarr.open(f"{argmax_task.task_name}/data.zarr/out_array")[:, 0, 0])
```

output:

```
simple-argmax ✔: 100%|███████████| 8/8 [00:00<00:00, 265.34blocks/s, ⧗=0, ▶=0, ✔=8, ✗=0, ∅=0]

Execution Summary
-----------------

  Task simple-argmax:

    num blocks : 8
    completed ✔: 8 (skipped 0)
    failed    ✗: 0
    orphaned  ∅: 0

    all blocks processed successfully
[[5. 0. 9. 0. 8. 9. 2. 4. 3. 4.]
 [9. 0. 0. 4. 9. 9. 4. 0. 5. 3.]
 [3. 6. 8. 9. 8. 0. 6. 5. 4. 7.]]
[1. 2. 0. 2. 1. 0. 2. 2. 1. 2.]
```
