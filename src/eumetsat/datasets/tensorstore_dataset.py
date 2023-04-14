import asyncio
import os

import tensorstore as ts
from jaxtyping import Int, UInt8, Array

from eumetsat.datasets.abc_dataset import BaseDataset
from eumetsat.datasets.utils import FileNameProps


class EMTensorstoreDataset(BaseDataset):
    @property
    def props(self) -> FileNameProps:
        return self._props

    def __init__(self, path: str):
        f_name = os.path.basename(path)
        self._props = FileNameProps.from_str(f_name)
        self._imgs = ts.open(
            {
                'driver': "n5",
                'kvstore': {
                    'driver': 'file',
                    'path': str(path),
                },
            },
            read=True,
            write=False
        ).result()

    def __len__(self):
        return self._imgs.shape[0]

    def ts_to_idx(self, ts: int) -> int:
        return ts - self.props.time_zero

    async def a_batch_from_timesamps_idx(self, ts_index: Int[Array, "batch"]) -> UInt8[Array, "batch 500 500 12"]:
        ts_index = (ts_index - self.props.time_zero) // self.props.freq
        return await self._imgs[ts_index].read()

    def batch_from_timesamps_idx(self, ts_index: Int[Array, "batch"]) -> UInt8[Array, "batch 500 500 12"]:
        task = self.a_batch_from_timesamps_idx(ts_index)
        result = asyncio.run(task)
        return result
