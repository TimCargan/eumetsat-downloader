import asyncio
import queue
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import serde.json
import tensorstore
from absl import flags, app, logging
from rich.progress import Progress

from eumetsat import IMG_LAYERS
from eumetsat.datasets.utils import Metadata, FileNameProps
from hemera.path_translator import get_path

flags.DEFINE_string("min_date", default="2018-01-01 00:00", help="Min date for image files")
flags.DEFINE_string("max_date", default="2018-01-04 23:59", help="Max date for image files")
flags.DEFINE_string("png_meta", default="png_metadata.json", help="Name of the png metadata")
flags.DEFINE_integer("freq", default=3600, help="Freq (in seconds) of the images")
flags.DEFINE_boolean("save", default=True, help="Save the dataset")

FLAGS = flags.FLAGS


def read(kv: tuple[int, str], z: int, freq: int, img_base_path: Path, img_array: np.ndarray = None,
         offset: int = 0) -> np.ndarray:
    k, v = kv
    t_path = img_base_path / v
    index = (k - z) // freq - offset if img_array is not None else 0
    img_array = np.zeros((1, 500, 500, 12), dtype=np.uint8) if img_array is None else img_array

    for i, l in enumerate(IMG_LAYERS):
        path = t_path / f"format={l}/img.png"
        with iio.imopen(path, "r") as img_file:
            img = img_file.read(index=0)[..., 0]
            img_array[index, ..., i] = img[:]
    return img_array


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def load_metadata(png_base_path: Path, metadata_name: str = "png_metadata.json") -> Metadata:
    meta_path = png_base_path / metadata_name
    with meta_path.open() as f:
        data = f.read()
    return serde.json.from_json(Metadata, data)


def validate_start(target_start: datetime, source_start: datetime, freq_min: int = 15) -> set[datetime]:
    if target_start < source_start:
        logging.warning("Target start datetime (%s) is before source start (%s)", target_start.isoformat(),
                        source_start.isoformat())

        missing = set(pd.date_range(target_start, source_start, freq=f"{freq_min}min"))
        return missing
    return set()


def validate_end(target_end: datetime, source_end: datetime, freq_min: int = 15) -> set[datetime]:
    if target_end > source_end:
        logging.warning("Target end datetime (%s) is after source end (%s)", target_end.isoformat(),
                        source_end.isoformat())
        missing = set(pd.date_range(source_end, target_end, freq=f"{freq_min}min"))
        return missing
    return set()

async def main(argv):
    ts_start = datetime.fromisoformat(FLAGS.min_date)
    ts_end = datetime.fromisoformat(FLAGS.max_date)

    # Load paths, using metadata
    base_path = get_path("data") / "EUMETSAT/UK-EXT"  # TODO: look at making this a param? can have raw / staging etc
    source_meta = load_metadata(base_path, FLAGS.png_meta)
    img_base_path = source_meta.data_location

    freq = FLAGS.freq
    freq_min = freq // 60

    target_date_range = pd.date_range(ts_start, ts_end, freq=f"{freq_min}min")

    lead_missing = validate_start(ts_start, source_meta.first_example_date, freq_min=freq_min)
    end_missing = validate_end(ts_end, source_meta.last_example_date, freq_min=freq_min)
    not_in_source = lead_missing | end_missing
    known_missing = set(target_date_range) & source_meta.missing
    expected_missing = not_in_source | known_missing

    date_dict = {int(ts.timestamp()): ts.strftime("year=%Y/month=%m/day=%d/time=%H_%M") for ts in target_date_range}
    samples = len(date_dict)
    z = int(target_date_range[0].timestamp())

    logging.info("will scan for %d samples", samples)
    logging.info("Known missing %s", sorted(expected_missing))

    fn = FileNameProps(time_zero=ts_start, time_end=ts_end, freq=freq)
    out_path = get_path("data") / f"EUMETSAT/UK-EXT/{fn}.ts.zarr"

    dataset = tensorstore.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': str(out_path),
        },
        'schema': {
            'dtype': 'uint8',
            'domain': {
                "rank": 4,
                "shape": [samples, 500, 500, 12],
                "labels": ["ts", "h", "w", "c"],
            },
            "dimension_units": [[freq, "s"], [13 / 500, "deg"], [17 / 500, "deg"], ""],
        },
        'metadata': {
            'compression': {
                'type': 'blosc',
                "cname": "zlib",
                "clevel": 9,
                "shuffle": 2
            },
            'dataType': 'uint8',
            'blockSize': [24, 128, 128, 12],
        },
        'create': True,
        'delete_existing': True
    }).result()

    writes = []
    write_q = queue.Queue(maxsize=500)
    chunk_size = 24
    d = list(date_dict.items())
    d = chunker(d, chunk_size)

    p = Progress()
    p.start()
    read_bar = p.add_task("[dark_orange]Reading", total=len(date_dict))
    write_bar = p.add_task("[green]Writing", total=len(date_dict) // chunk_size)

    def cb(f):
        write_q.get()
        write_q.task_done()
        p.update(write_bar, advance=1)

    for i, kvc in enumerate(d):
        r_chunk_size = len(kvc)
        data = np.zeros((r_chunk_size, 500, 500, 12), dtype=np.uint8)
        for x, kv in enumerate(kvc):
            dt = datetime.utcfromtimestamp(kv[0])
            if dt not in expected_missing:
                try:
                    read(kv, z=z, freq=freq, img_array=data, offset=i * chunk_size, img_base_path=img_base_path)
                except FileNotFoundError as e:
                    print(f"Tried to read {dt}.... not sure why ({e})")
            p.update(read_bar, advance=1)

        s = i * chunk_size
        e = s + r_chunk_size
        write_future = dataset[s:e].write(data)
        write_future.add_done_callback(cb)
        write_q.put(write_future)
        writes.append(write_future)

    print("Queue join")
    write_q.join()
    print("Async wait")
    await asyncio.gather(*writes)
    p.stop()

    logging.info("Writing meta")
    metadata = Metadata(
        metadata_created=datetime.now(),
        first_example_date=ts_start,
        last_example_date=ts_end,
        example_count=len(set(target_date_range) - expected_missing),
        missing=expected_missing,
        freq_seconds=freq,
        data_source=Path(img_base_path),
        data_location=out_path
    )

    metadata_str = serde.json.to_json(metadata)
    with (out_path/"img_meta.json").open("w") as f:
        f.write(metadata_str)
    logging.info("done")


def amain(var):
    asyncio.run(main(var))


if __name__ == "__main__":
    app.run(amain)
