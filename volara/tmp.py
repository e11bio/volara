"""Deprecated alias for :mod:`volara.segment_utils`.

This module was renamed to :mod:`volara.segment_utils` because ``tmp`` implied
scratch code, while the contents are a load-bearing part of the public API
(``seg_to_affgraph`` appears in the published tutorial, and the numba kernels are
used throughout :mod:`volara.blockwise`).

Importing this module still works but emits a :class:`DeprecationWarning`.
Update imports to::

    from volara.segment_utils import replace_values, seg_to_affgraph
"""

import warnings

from volara.segment_utils import (
    filter_mapping_to_block,
    prepare_mapping,
    replace_values,
    replace_values_sorted,
    seg_to_affgraph,
    warmup_replace_values_sorted,
)

__all__ = [
    "filter_mapping_to_block",
    "prepare_mapping",
    "replace_values",
    "replace_values_sorted",
    "seg_to_affgraph",
    "warmup_replace_values_sorted",
]

warnings.warn(
    "volara.tmp has been renamed to volara.segment_utils. "
    "Importing from volara.tmp is deprecated and the alias will be removed in a "
    "future release; import from volara.segment_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
