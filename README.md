[![tests](https://github.com/pattonw/volara/actions/workflows/tests.yaml/badge.svg)](https://github.com/pattonw/volara/actions/workflows/tests.yaml)
[![black](https://github.com/pattonw/volara/actions/workflows/black.yaml/badge.svg)](https://github.com/pattonw/volara/actions/workflows/black.yaml)
[![mypy](https://github.com/pattonw/volara/actions/workflows/mypy.yaml/badge.svg)](https://github.com/pattonw/volara/actions/workflows/mypy.yaml)

# volara
Easy application of common blockwise operations for image processing of arbitrarily large volumetric microscopy.

# Available blockwise operations:
- `Predict`: Model Prediction
- `FragmentExtraction`: Fragment extraction via mutex watershed
- `AffAgglom`: Supervoxel affinity score edge creation
- `ArgMax`: Argmax accross predicted probabilities
- `DistanceAgglom`: Supervoxel distance score edge creation
- `GlobalSeg`: Global creation of look up tables for fragment -> segment agglomeration
- `LUT`: Remapping and saving fragments as segments
- `PsuedoAffs`: Create "psueodo affinities" from distance measures (e.g. cosine similarity on LSD predictions)
- `SeededExtractFrags`: Fragment extraction via mutex watershed that accepts skeletonized seed points for constrained fragment extraction
