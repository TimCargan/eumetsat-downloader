import os

import numpy as np
from jaxtyping import Int, UInt8, Array


class NumpyImageBlob:
    def __init__(self, path: str):
        self._imgs = np.load(path, mmap_mode='r')
        f_name = os.path.basename(path)
        # f"imgs.z:{z},f:{freq}min" pattern
        self.ts_zero = int(f_name.split(",")[0].split("=")[1])
        self.freq = int(f_name.split(",")[1].split("=")[1][:-7])

    def __len__(self):
        return self._imgs.shape[0]

    def ts_to_idx(self, ts: int) -> int:
        return ts - self.ts_zero

    def batch_from_timesamps_idx(self, ts_index: Int[Array, "batch"]) -> UInt8[Array, "batch 500 500 12"]:
        ts_index = (ts_index - self.ts_zero) // self.freq
        return self._imgs[ts_index]
