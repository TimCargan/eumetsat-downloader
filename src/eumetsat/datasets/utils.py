from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import serde


@serde.serde
@dataclass
class Metadata:
    metadata_created: datetime
    data_source: Path
    data_location: Path

    first_example_date: datetime
    last_example_date: datetime

    example_count: int
    missing: set[datetime]

    freq_seconds: int




@dataclass
class FileNameProps:
    time_zero: datetime = None
    time_end: datetime = None
    freq: int = None

    @classmethod
    def from_str(cls, name:str) -> FileNameProps:
        "Name format is `img_z=2020T00,e=2020T00,f=000.xyz"
        name_kv = name.split("_")[-1].split(".")[0] # split out the _ and .
        name_kv = name_kv.split(",") #make it a list of kv
        name_kv = dict([(*kv.split("="),) for kv in name_kv])

        ts_zero = str(name_kv["z"]).replace("_", ":")
        ts_end = str(name_kv["e"]).replace("_", ":")
        freq = int(name_kv["f"])

        ts_zero = datetime.fromisoformat(ts_zero)
        ts_end = datetime.fromisoformat(ts_end)
        return cls(time_zero=ts_zero, time_end=ts_end, freq=freq)

    @property
    def file_name(self) -> str:
        z = self.time_zero.isoformat().replace(":", "_")
        e = self.time_end.isoformat().replace(":", "_")
        return f"img_z={z},e={e},f={self.freq}"

    def __str__(self) -> str:
        return self.file_name
