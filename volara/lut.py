from pathlib import Path

import numpy as np

from .utils import StrictBaseModel


class LUT(StrictBaseModel):
    """
    A class for defining look up tables
    """

    path: Path | str
    """
    The path at which we will read/write the look up table
    """

    @property
    def name(self) -> str:
        return self.file.stem

    @property
    def file(self) -> Path:
        if isinstance(self.path, str):
            return (
                Path(self.path)
                if self.path.endswith(".npz")
                else Path(f"{self.path}.npz")
            )
        elif isinstance(self.path, Path):
            return self.path

    def drop(self):
        if self.file.exists():
            self.file.unlink()

    def save(self, lut, edges=None):
        np.savez_compressed(self.file, fragment_segment_lut=lut, edges=edges)

    def load(self) -> np.ndarray:
        return np.load(self.file)["fragment_segment_lut"]
