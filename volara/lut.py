import io
from collections.abc import Sequence

import numpy as np
from pydantic import AliasChoices, Field, model_validator
from upath import UPath

from volara.segment_utils import replace_values

from .paths import PydanticUPath, is_local, rm, to_upath
from .utils import StrictBaseModel


class LUT(StrictBaseModel):
    """
    A class for defining look up tables
    """

    path: PydanticUPath
    """
    The path at which we will read/write the look up table: a local path
    (`/path/to/data.zarr/lut.npz`) or a URL on any fsspec filesystem
    (`s3://bucket/path/to/data.zarr/lut.npz`). The `.npz` extension is
    appended if it is missing.
    """

    storage_options: dict | None = Field(
        default=None, validation_alias=AliasChoices("storage_options", "s3_kwargs")
    )
    """
    fsspec filesystem options for a remote `path` (credentials, a custom
    endpoint such as Lyve Cloud, ...). `None` uses the environment.
    `s3_kwargs` is accepted as the former name.
    """

    @model_validator(mode="after")
    def _apply_storage_options(self):
        if self.storage_options:
            self.path = to_upath(self.path, **self.storage_options)
        return self

    @property
    def is_local(self) -> bool:
        return is_local(self.path)

    @property
    def file(self) -> UPath:
        """The location of this look up table, with the `.npz` suffix."""
        return (
            self.path
            if self.path.suffix == ".npz"
            else self.path.with_name(f"{self.path.name}.npz")
        )

    @property
    def uri(self) -> str:
        """The normalized location of this look up table as a string."""
        return str(self.file)

    @property
    def name(self) -> str:
        return self.file.stem

    def exists(self) -> bool:
        return self.file.exists()

    def drop(self):
        rm(self.file, recursive=False)

    def save(self, lut: np.ndarray, edges=None):
        arrays = {"fragment_segment_lut": lut.astype(int)}
        if edges is not None:
            arrays["edges"] = edges

        with self.file.open("wb") as f:
            np.savez_compressed(f, **arrays)
        if not self.is_local:
            invalidate = getattr(self.file.fs, "invalidate_cache", None)
            if invalidate is not None:
                invalidate(self.file.path)

    def load(self) -> np.ndarray | None:
        if not self.file.exists():
            return None
        with self.file.open("rb") as f:
            buffer = io.BytesIO(f.read())
        with np.load(buffer) as data:
            return data["fragment_segment_lut"]

    def __add__(self, other):
        """
        Add two disjoint LUTs together. See `LUTS.load` for concatenation of
        disjoint mappings i.e. {0:1} + {2:3} = {0:1, 2:3}, and
        `LUTS.load_iterated` for chaining mappings i.e. {0:1} + {1:2} = {0:2}.
        """
        if isinstance(other, LUT):
            return LUTS(luts=[self, other])
        raise TypeError(f"Cannot add {type(other)} to LUT")


class LUTS:
    def __init__(self, luts: LUT | Sequence[LUT]):
        self.luts: Sequence[LUT] = luts if not isinstance(luts, LUT) else [luts]

    def __add__(self, other):
        if isinstance(other, LUTS):
            return LUTS(list(self.luts) + list(other.luts))
        elif isinstance(other, LUT):
            return LUTS(list(self.luts) + [other])
        raise TypeError(f"Cannot add {type(other)} to LUTS")

    def load(self):
        mappings = (lut.load() for lut in self.luts)
        return np.concatenate(
            [mapping for mapping in mappings if mapping is not None], axis=1
        )  # type: ignore

    def load_iterated(self):
        starting_map = self.luts[0].load()
        assert starting_map is not None, "No lookup tables to load"
        for lut in self.luts[1:]:
            next_map = lut.load()
            if next_map is not None:
                starting_map[1] = replace_values(
                    starting_map[1], next_map[0], next_map[1]
                )
        return starting_map
