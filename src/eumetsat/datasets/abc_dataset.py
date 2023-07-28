from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from jaxtyping import Array, Int, UInt8

if TYPE_CHECKING:
    from eumetsat.datasets.utils import FileNameProps


class BaseDataset(abc.ABC):

    @property
    @abc.abstractmethod
    def props(self) -> FileNameProps:
        pass

    @props.setter
    def props(self, value):
        pass

    def ts_to_idx(self, ts: int) -> int:
        return ts - int(self.props.time_zero.timestamp())

    @abc.abstractmethod
    def batch_from_timesamps_idx(self, ts_index: Int[Array, "batch"]) -> UInt8[Array, "batch 500 500 12"]:
        pass
