[![tests](https://github.com/e11bio/volara/actions/workflows/tests.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/tests.yaml)
[![ruff](https://github.com/e11bio/volara/actions/workflows/ruff.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/ruff.yaml)
[![mypy](https://github.com/e11bio/volara/actions/workflows/mypy.yaml/badge.svg)](https://github.com/e11bio/volara/actions/workflows/mypy.yaml)
<!-- [![codecov](https://codecov.io/gh/e11bio/volara/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/e11bio/volara) -->

# volara
Easy application of common blockwise operations for image processing of arbitrarily large volumetric microscopy.

# Available blockwise operations:
- `FragmentExtraction`: Fragment extraction via mutex watershed
- `AffAgglom`: Supervoxel affinity score edge creation
- `ArgMax`: Argmax accross predicted probabilities
- `DistanceAgglom`: Supervoxel distance score edge creation
- `GlobalSeg`: Global creation of look up tables for fragment -> segment agglomeration
- `LUT`: Remapping and saving fragments as segments
- `SeededExtractFrags`: Fragment extraction via mutex watershed that accepts skeletonized seed points for constrained fragment extraction
