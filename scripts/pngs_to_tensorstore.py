"""Script to convert PNG dataset to a tensorstore dataset.

This script loads the PNGs and creates / updates a tensorstore dataset with them

Params:
    min_date: the start date to look for pngs (will have all zeros out bounds of the pngs)
    max_date: the end data to look for pngs (will have all zeros out bounds of the pngs)
    png_meta: name of the metadata file, made by running the png_to_meta script. It gives hits for where missing files are etc.
    freq: Frequency in seconds of the images (usually 900 for 15min, or 3600 for 1 hour)
"""
import queue
import threading
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import serde.json
import tensorstore
from absl import app, flags, logging
from rich.progress import DownloadColumn, Progress, TimeElapsedColumn

from eumetsat.datasets.utils import FileNameProps, Metadata, load_metadata, read_png
from hemera.path_translator import get_path
from hemera.standard_logger import logging

flags.DEFINE_string("min_date", default="2018-01-01 00:00", help="Min date for image files")
flags.DEFINE_string("max_date", default="2018-01-04 23:59", help="Max date for image files")
flags.DEFINE_string("png_meta", default="png_metadata.json", help="Name of the png metadata")
flags.DEFINE_integer("freq", default=3600, help="Freq (in seconds) of the images")
flags.DEFINE_boolean("overwrite", default=False, help="Overwrite existing file")
FLAGS = flags.FLAGS
END_MSG = None
TS_BYTES = 12 * 500 * 500


class Writer(threading.Thread):
    """Threaded Writer.

    Class to manage threaded writing and callbacks to update progress bars.
    """
    def __init__(self, write_q: queue.Queue, dataset: tensorstore.TensorStore, p_context: Progress, write_bar,
                 copy_bar):
        """Create Writer Object

        Args:
            write_q: Queue to put data to be written to (tuple of [datetime, chunk_size, chunk (ndarray), slice])
            dataset: Tensorstore dataset
            p_context: progress bar context (rich Progress context)
            write_bar: write progress bar
            copy_bar: copy progress bar
        """
        super().__init__(daemon=True)
        self.write_q = write_q
        self.dataset = dataset
        self.pb_queue = queue.Queue()
        self.p_context = p_context
        self.copy_bar = copy_bar
        self.write_bar = write_bar

    def update(self, bar, step):
        """Update given progress bar.

        Args:
            bar: Progress bar to update
            step: Number of steps to update bar
        """
        self.p_context.update(bar, advance=step)

    def _write_update(self, future: tensorstore.Future):
        """Write complete callback.

        Updates the queues and write progress bar after a write has been completed.
        """
        r_chunk_size = self.pb_queue.get(timeout=1)
        r_chunk_size = r_chunk_size * TS_BYTES
        self.update(self.write_bar, r_chunk_size)
        self.write_q.task_done()
        self.pb_queue.task_done()

    def _copy_update(self, size, bar):
        """Copy callback. Updates the progress bar."""
        r_chunk_size = size * TS_BYTES
        self.update(bar, r_chunk_size)

    def run(self):
        """Main thread for the writer

        Reads from the write queue until an END_MSG is received.
        for each data chunk to write
            Submits the data to be written by the tensorstore object (that deals with all the IO),
            Adds the callbacks to the future object:
                1. copy (load into the tensorstore memory object)
                2. write completions (committed chunk to disk)
            Adds the future object to a list to stop python GC

        One all writes are submitted it prints some telemetry and joins the queues blocking until all tasks are done
        """
        writes = []
        while True:
            msg = self.write_q.get()
            if msg is END_MSG:
                self.write_q.task_done()
                break

            dt, chunk_size, chunk, sli = msg
            logging.debug("Writing %s, %s, %s", dt, chunk.shape, id(chunk))

            write_future = self.dataset[sli].write(chunk)
            self.pb_queue.put(chunk_size)
            write_future.copy.add_done_callback(lambda x: self._copy_update(chunk_size, self.copy_bar))
            write_future.commit.add_done_callback(self._write_update)
            writes.append(write_future)

        logging.info("Reader queued all tasks [%s, %s]", f"{self.write_q.unfinished_tasks=}", f"{self.pb_queue.unfinished_tasks=}")
        writes_done = reduce(lambda a, b: a + int(b.done()), writes, 0)
        logging.info("Reader: done %d", writes_done)

        self.pb_queue.join()
        logging.info("Reader exit, [%s, %s]", f"{self.write_q.unfinished_tasks=}", f"{self.pb_queue.unfinished_tasks=}")


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


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


def get_spec(path: Path, samples: int, freq: int) -> dict:
    create = not path.exists() or FLAGS.overwrite
    overwrite = FLAGS.overwrite if path.exists() else False
    return {
        'driver': 'n5',
        'context': {
            "cache_pool": {"total_bytes_limit": 10_000_000},
            "cache_pool#remote": {"total_bytes_limit": 10_000_000},
            "data_copy_concurrency": {"limit": 8},
            "file_io_concurrency": {"limit": 8}
        },
        'kvstore': {
            'driver': 'file',
            'path': str(path),
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
                "cname": "blosclz",
                "clevel": 9,
                "shuffle": 2
            },
            'dataType': 'uint8',
            'blockSize': [24, 250, 250, 12],
        },
        "open": not overwrite,
        "create": create,
        "delete_existing": overwrite
    }


def main(argv):
    # Load paths, using metadata
    base_path = get_path("data") / "EUMETSAT/UK-EXT"  # TODO: look at making this a param? can have raw / staging etc
    source_meta = load_metadata(base_path, FLAGS.png_meta)
    img_base_path = source_meta.data_location

    # Load time range to write out
    freq = FLAGS.freq
    freq_min = freq // 60

    ts_start = datetime.fromisoformat(FLAGS.min_date)
    ts_end = datetime.fromisoformat(FLAGS.max_date)

    target_date_range = pd.date_range(ts_start, ts_end, freq=f"{freq_min}min")

    # Calculate the set of expected missing images, use meta to fund the lower and upper bounds of existing
    lead_missing = validate_start(ts_start, source_meta.first_example_date, freq_min=freq_min)
    end_missing = validate_end(ts_end, source_meta.last_example_date, freq_min=freq_min)
    not_in_source = lead_missing | end_missing
    known_missing = set(target_date_range) & source_meta.missing
    expected_missing = not_in_source | known_missing

    # Generate the dict of images to load
    date_dict = {int(ts.timestamp()): ts.strftime("year=%Y/month=%m/day=%d/time=%H_%M") for ts in target_date_range}
    samples = len(date_dict)
    z = int(target_date_range[0].timestamp())

    logging.info("Will scan for %d samples", samples)
    logging.debug("Known missing %s", sorted(expected_missing))

    # Create the tensorstore dataset (using the standard naming format)
    fn = FileNameProps(time_zero=ts_start, time_end=ts_end, freq=freq)
    out_path = get_path("data") / f"EUMETSAT/UK-EXT/{fn}.ts.zarr"
    dataset = tensorstore.open(get_spec(out_path, samples, freq)).result()

    # Set up progress bar
    p = Progress(*Progress.get_default_columns(), TimeElapsedColumn(), DownloadColumn())
    read_bar = p.add_task("[dark_orange]Reading", total=len(date_dict) * TS_BYTES)
    copy_bar = p.add_task("[dark_orange]TS-copy", total=len(date_dict) * TS_BYTES)
    write_bar = p.add_task("[green]Writing", total=len(date_dict) * TS_BYTES)
    write_q = queue.Queue(maxsize=2)
    p.start()

    # Create writer object
    writer = Writer(write_q=write_q, dataset=dataset, p_context=p, write_bar=write_bar, copy_bar=copy_bar)
    writer.start()

    # Chunk data for reading
    chunk_size = 24 * 4 * 4  # TODO: magic number, fix it
    d = chunker(list(date_dict.items()), chunk_size)

    # Enumerate over chunked windows of data reading them into a numpy object and added them to the write queue
    for i, kvc in enumerate(d):
        r_chunk_size = len(kvc)
        data = np.zeros((r_chunk_size, 500, 500, 12), dtype=np.uint8)

        # Read PNGS
        for x, kv in enumerate(kvc):
            dt = datetime.utcfromtimestamp(kv[0])
            if dt not in expected_missing:
                try:
                    read_png(kv, img_base_path=img_base_path, img_array=data, z=z, freq=freq, offset=i * chunk_size)
                except FileNotFoundError as e:
                    logging.error(f"Tried to read {dt}.... not sure why ({e})")
            p.update(read_bar, advance=1 * TS_BYTES)

        # Submit chunk to write
        s = i * chunk_size
        e = s + r_chunk_size
        write_q.put((dt, r_chunk_size, data, slice(s, e)))  # noqa

    # Add END_MSG and join queue
    write_q.put(END_MSG)
    logging.info("Queue join -- waiting for writes to finish curr size (%d)", write_q.unfinished_tasks)
    write_q.join()
    logging.info("Write queue exit")
    p.stop()

    # Create metadata file, do this last to indicate sucess
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

    with (out_path / "img_meta.json").open("w") as f:
        metadata_str = serde.json.to_json(metadata)
        f.write(metadata_str)
    logging.info("done")


if __name__ == "__main__":
    app.run(main)
