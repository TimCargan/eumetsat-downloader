import abc

from jaxtyping import Int, UInt8, Array

from eumetsat.datasets import FileNameProps


class BaseDataset(abc.ABC):

    @property
    @abc.abstractmethod
    def props(self) -> FileNameProps:
        pass

    @props.setter
    def props(self, value):
        pass
    def ts_to_idx(self, ts: int) -> int:
        return ts - self.props.time_zero

    @abc.abstractmethod
    def batch_from_timesamps_idx(self, ts_index: Int[Array, "batch"]) -> UInt8[Array, "batch 500 500 12"]:
        pass
