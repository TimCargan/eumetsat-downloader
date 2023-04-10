import os

import numpy as np
from jaxtyping import Int, UInt8, Array

from eumetsat.datasets import FileNameProps, BaseDataset


class NumpyImageBlob(BaseDataset):
    @property
    def props(self) -> FileNameProps:
        return self._props

    def __init__(self, path: str):
        f_name = os.path.basename(path)
        self._props = FileNameProps.from_str(f_name)
        self._imgs = np.load(path, mmap_mode='r')

    def __len__(self):
        return self._imgs.shape[0]

    def ts_to_idx(self, ts: int) -> int:
        return ts - self.props.time_zero

    def batch_from_timesamps_idx(self, ts_index: Int[Array, "batch"]) -> UInt8[Array, "batch 500 500 12"]:
        ts_index = (ts_index - self.props.time_zero) // self.props.freq
        return self._imgs[ts_index]
