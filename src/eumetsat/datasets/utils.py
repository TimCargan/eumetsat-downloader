from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import serde

from eumetsat import IMG_LAYERS


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

    def timestamp_to_idx(self, ts: int):
        ts_index = (ts - int(self.first_example_date.timestamp())) // self.freq_seconds
        return ts_index


def load_metadata(data_base_path: Path, metadata_name: str = "metadata.json") -> Metadata:
    """Load Metadata from json file.

    Args:
        data_base_path: Path to the folder where the data is
        metadata_name: Name of the metadata file

    Returns:
        A Metadata object
    """
    meta_path = data_base_path / metadata_name
    with meta_path.open() as f:
        data = f.read()
    return serde.json.from_json(Metadata, data)

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
        name_kv = dict([(kv.split("=")[0], kv.split("=")[1]) for kv in name_kv])

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


def read_png(kv: tuple[int, str], img_base_path: Path, img_array: np.ndarray = None, z: int = 0, freq: int = 3600, offset: int = 0) -> np.ndarray:
    """Read pngs into a ndarray

    Images are loaded into a numpy array, either given or if none a new one is created
    If an array is given, the images are loaded into the index as defined by the timestamp, freq, zero timestamp and offset:
         i(ts) = (ts - z) // freq - offset

    Args:
        kv: tuple of (int timestamp, string of filepath)
        img_base_path: base path for the PNGS
        img_array: numpy array to load the images into
        z: timestamp of the 'zero offset' image
        freq: frequency of images in the numpy array
        offset: index offset

    Returns:
        ndarray of all images channels for the given datetime loaded.
        The ndarray is channels last, ie [i(ts), x, y, c]
    """
    k, v = kv
    t_path = img_base_path / v

    # Calculate index or create ndarray
    if img_array is None:
        index = 0
        img_array = np.zeros((1, 500, 500, 12), dtype=np.uint8)
    else:
        index = (k - z) // freq - offset

    # Load image data from pngs
    for i, l in enumerate(IMG_LAYERS):
        path = t_path / f"format={l}/img.png"
        with iio.imopen(path, "r") as img_file:
            img = img_file.read(index=0)[..., 0]
            img_array[index, ..., i] = img[:]

    return img_array
