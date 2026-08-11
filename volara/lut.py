import io
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import numpy as np

from volara.segment_utils import replace_values

from .utils import StrictBaseModel


class LUT(StrictBaseModel):
    """
    A class for defining look up tables
    """

    path: Path | str
    """
    The path at which we will read/write the look up table. Either a local
    path (`/path/to/data.zarr/lut.npz`) or an s3 uri
    (`s3://bucket/path/to/data.zarr/lut.npz`). The `.npz` extension will be
    appended if it is missing from a string path.
    """

    s3_kwargs: dict | None = None
    """
    Optional config for S3-compatible stores like Lyve Cloud.
    Leave as None for standard S3 behavior (credentials from the environment).
    Only used when `path` is an `s3://` uri.
    """

    @property
    def is_s3(self) -> bool:
        """
        Whether this look up table lives in an s3 bucket rather than on
        the local filesystem.
        """
        return isinstance(self.path, str) and self.path.startswith("s3://")

    def _s3fs(self):
        import s3fs  # type: ignore[unresolved-import]

        if self.s3_kwargs is None:
            return s3fs.S3FileSystem()

        return s3fs.S3FileSystem(**self.s3_kwargs)

    @property
    def uri(self) -> str:
        """
        The normalized location of this look up table as a string. This is
        the only accessor that works for both local and s3 look up tables.
        """
        if self.is_s3:
            path = str(self.path)
            return path if path.endswith(".npz") else f"{path}.npz"
        return str(self.file)

    @property
    def name(self) -> str:
        if self.is_s3:
            return PurePosixPath(self.uri).stem
        return self.file.stem

    @property
    def file(self) -> Path:
        if isinstance(self.path, str):
            if self.is_s3:
                raise ValueError(
                    f"{self.path} is an s3 uri and has no local path, use `uri` instead"
                )
            return (
                Path(self.path)
                if self.path.endswith(".npz")
                else Path(f"{self.path}.npz")
            )
        elif isinstance(self.path, Path):
            return self.path
        else:
            raise TypeError(f"Invalid type for path ({self.path}): {type(self.path)}")

    def exists(self) -> bool:
        if self.is_s3:
            return self._s3fs().exists(self.uri)
        return self.file.exists()

    def drop(self):
        if self.is_s3:
            fs = self._s3fs()
            try:
                fs.rm(self.uri)
            except FileNotFoundError:
                pass
            fs.invalidate_cache(self.uri)
        elif self.file.exists():
            self.file.unlink()

    def save(self, lut: np.ndarray, edges=None):
        arrays = {"fragment_segment_lut": lut.astype(int)}
        if edges is not None:
            arrays["edges"] = edges

        if self.is_s3:
            fs = self._s3fs()
            with fs.open(self.uri, "wb") as f:
                np.savez_compressed(f, **arrays)
            fs.invalidate_cache(self.uri)
        else:
            np.savez_compressed(self.file, **arrays)

    def load(self) -> np.ndarray | None:
        if self.is_s3:
            fs = self._s3fs()
            if not fs.exists(self.uri):
                return None
            with fs.open(self.uri, "rb") as f:
                buffer = io.BytesIO(f.read())
            with np.load(buffer) as data:
                return data["fragment_segment_lut"]

        if not self.file.exists():
            return None
        with np.load(self.file) as data:
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
