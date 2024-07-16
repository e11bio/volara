from typing import Iterable, Union

import numpy as np
from numpy.typing import NDArray


def replace_values(
    labels: NDArray[np.int_], a: Union[Iterable[int], int], b: Union[Iterable[int]]
) -> NDArray[np.int_]:
    if isinstance(a, int):
        a = iter([a])
    else:
        a = iter(a)
    if isinstance(b, int):
        b = iter([b])
    else:
        b = iter(b)
    for u, v in zip(a, b):
        labels[labels == u] = v
    return labels
