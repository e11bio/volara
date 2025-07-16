from pathlib import Path

import numpy as np
from pydantic import field_validator

from .utils import StrictBaseModel


class LUT(StrictBaseModel):
    """
    A class for defining look up tables
    """

    path: Path
    """
    The path at which we will read/write the look up table
    """

    @field_validator("store", mode="after")
    @classmethod
    def path_path(cls, v) -> Path:
        try:
            if isinstance(v, str):
                return Path(v) if v.endswith(".npz") else Path(f"{v}.npz")
            else:
                return Path(v)
        except TypeError:
            raise ValueError(f"Invalid store path: {v}. Must be a path-like object.")

    @property
    def name(self) -> str:
        return self.path.stem

    def drop(self):
        if self.path.exists():
            self.path.unlink()

    def spoof(self, spoof_dir: Path):
        return self.__class__(path=spoof_dir / self.path.name)

    def save(self, lut, edges=None):
        np.savez_compressed(self.path, fragment_segment_lut=lut, edges=edges)

    def load(self) -> np.ndarray:
        return np.load(self.path)["fragment_segment_lut"]
