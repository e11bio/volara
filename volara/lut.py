from pathlib import Path
from collections.abc import Sequence

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

    def save(self, lut: np.ndarray, edges=None):
        np.savez_compressed(self.file, fragment_segment_lut=lut.astype(int), edges=edges)

    def load(self) -> np.ndarray | None:
        if not self.file.exists():
            return None
        return np.load(self.file)["fragment_segment_lut"]

    def __add__(self, other):
        """
        Add two disjoint LUTs together via simple concatenation.
        i.e. {0:1} + {1:2} = {0:2}
        """
        if isinstance(other, LUT):
            return LUTS(luts=[self, other])
        raise TypeError(f"Cannot add {type(other)} to LUT")


class LUTS:
    def __init__(self, luts: LUT | Sequence[LUT]):
        self.luts = luts if not isinstance(luts, LUT) else [luts]

    def __add__(self, other):
        if isinstance(other, LUTS):
            return LUTS(self.luts + other.luts)
        elif isinstance(other, LUT):
            return LUTS(self.luts + [other])
        raise TypeError(f"Cannot add {type(other)} to LUTS")

    def load(self):
        return np.concatenate(
            [lut.load() for lut in self.luts if lut.load() is not None], axis=1
        )

    def load_iterated(self):
        def merge_mapping(mapping1, mapping2):
            # Step 1: Union of first rows (source labels)
            combined_keys = np.union1d(mapping1[0], mapping2[0])

            # Step 2: Create dicts for fast lookup
            dict1 = dict(zip(mapping1[0], mapping1[1]))
            dict2 = dict(zip(mapping2[0], mapping2[1]))

            # Step 3: Build the new mapping values
            combined_values = []
            for key in combined_keys:
                if key in dict1:
                    combined_values.append(dict1[key])
                else:
                    combined_values.append(dict2[key])

            return np.array([combined_keys, combined_values])

        starting_map = self.luts[0].load()
        for lut in self.luts[1:]:
            next_map = lut.load()
            if next_map is not None:
                starting_map = merge_mapping(starting_map, next_map)
        return starting_map
