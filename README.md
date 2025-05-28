[![tests](https://github.com/e11bio/volara/actions/workflows/tests.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/tests.yaml)
[![ruff](https://github.com/e11bio/volara/actions/workflows/ruff.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/ruff.yaml)
[![mypy](https://github.com/e11bio/volara/actions/workflows/mypy.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/mypy.yaml)
<!-- [![codecov](https://codecov.io/gh/e11bio/volara/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/e11bio/volara) -->

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

# Useful links
- [API Reference](https://e11bio.github.io/volara/api.html)
- [Basic tutorial](https://e11bio.github.io/volara/tutorial.html)
- [Cremi inference tutorial](https://e11bio.github.io/volara-torch/examples/cremi/cremi.html)
- [Cremi affinity agglomeration tutorial](https://e11bio.github.io/volara/examples/cremi/cremi.html)
- [Building a custom task](https://e11bio.github.io/volara/examples/getting_started/basics.html)

# Architecture
![](https://github.com/e11bio/volara/blob/main/docs/source/_static/Diagram-transparent%20bg2.png)
This diagram visualizes the lifetime of a block in volara. On the left we are reading array and/or graph data with optional padding for a specific block. This data is then processed, and written to the output on the right. For every block processed we also mark it done in a separate Zarr. Once each worker completes a block, it will fetch the next. This process continues until the full input dataset has been processed.

# Available blockwise operations:
- `ExtractFrags`: Fragment extraction via mutex watershed
- `AffAgglom`: Supervoxel affinity score edge creation
- `GraphMWS`: Global creation of look up tables for fragment -> segment agglomeration
- `Relabel`: Remapping and saving fragments as segments
- `SeededExtractFrags`: Fragment extraction via mutex watershed that accepts skeletonized seed points for constrained fragment extraction
- `ArgMax`: Argmax accross predicted probabilities
- `DistanceAgglom`: Supervoxel distance score edge creation. Computed between stored supervoxel embeddings. 
- `ComputeShift`: Compute shift between moving and fixed image using phase cross correlation
- `ApplyShift`: Apply computed shift to register moving image to fixed image
- `Threshold`: Intensity threshold an array
