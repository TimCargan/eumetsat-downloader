from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FileNameProps:
    name: str = None
    time_zero: int = None
    time_end: int = None
    freq: int = None

    @classmethod
    def from_str(cls, name:str) -> FileNameProps:
        "Name format is `img_z=000,e=000,f=000.xyz"
        name_kv = name.split("_")[-1].split(".")[0] # split out the _ and .
        name_kv = name_kv.split(",") #make it a list of kv
        name_kv = dict([(*kv.split("="),) for kv in name_kv])

        ts_zero = int(name_kv["z"])
        ts_end = int(name_kv["e"])
        freq = int(name_kv["f"])

        return cls(name=name, time_zero=ts_zero, time_end=ts_end, freq=freq)

