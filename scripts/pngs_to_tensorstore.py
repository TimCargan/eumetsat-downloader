import asyncio
import os
import os.path
from datetime import datetime

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tensorstore
from absl import flags, app, logging
from hemera.path_translator import get_path

from eumetsat import IMG_LAYERS

flags.DEFINE_string("min_date", default="2018-01-01 00:00", help="Min date for image files")
flags.DEFINE_string("max_date", default="2018-01-04 23:59", help="Max date for image files")
flags.DEFINE_integer("freq", default=900, help="Numpy data blob")
flags.DEFINE_boolean("save", default=True, help="Save the dataset")

FLAGS = flags.FLAGS


def read(kv, z, freq, img_array=None, offset=0):
    k, v = kv
    img_base_path = os.path.join(get_path("data"), "EUMETSAT/UK-EXT")
    t_path = img_base_path + v
    logging.debug(f"Reading {t_path}")
    index = (k - z) // freq - offset if img_array is not None else 0

    img_array = np.zeros((1, 500, 500, 12), dtype=np.uint8) if img_array is None else img_array

    for i, l in enumerate(IMG_LAYERS):
        path = t_path + f"/format={l}/img.png"
        if not os.path.exists(path):
            logging.warning(f"File not found {path}")
            return
        with iio.imopen(path, "r") as img_file:
            img = img_file.read(index=0)[..., 0]
            img_array[index, ..., i] = img
    return img_array

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

async def main(argv):
    ts_start = datetime.fromisoformat(FLAGS.min_date)
    ts_end = datetime.fromisoformat(FLAGS.max_date)

    img_base_path = os.path.join(get_path("data"), "EUMETSAT/UK-EXT")
    freq = FLAGS.freq
    freq_min = freq // 60

    date_rage = pd.date_range(ts_start, ts_end, freq=f"{freq_min}min")
    date_dict = {int(ts.timestamp()): ts.strftime("/year=%Y/month=%m/day=%d/time=%H_%M") for ts in date_rage}
    samples = len(date_dict)
    print(samples)
    z = int(date_rage[0].timestamp())

    out_path = os.path.join(get_path("data"),
                        f"EUMETSAT/UK-EXT/img_z={int(ts_start.timestamp())},e={int(ts_end.timestamp())},f={freq}.ts.zarr")

    dataset = tensorstore.open({
        'driver': 'n5',
         'kvstore': {
             'driver': 'file',
             'path': out_path,
         },
        'schema': {
            'dtype': 'uint8',
            'domain': {
                "rank": 4,
                "shape": [samples, 500, 500, 12],
                "labels": ["ts", "h", "w", "c"],
            },
            "dimension_units": [[freq, "s"], [13/500, "deg"], [17/500, "deg"], ""],
        },
         'metadata': {
             'compression': {
                 'type': 'blosc',
                 "cname": "zlib",
                 "clevel": 9,
                 "shuffle": 2
             },
             'dataType': 'uint8',
             'blockSize': [24, 32, 32, 12],
         },
         'create': True,
         'delete_existing': True
     }).result()
    writes = []

    chunk_size = 24
    d = list(date_dict.items())
    d = chunker(d, chunk_size)

    for i, kvc in enumerate(d):
        data = np.zeros((chunk_size, 500, 500, 12), dtype=np.uint8)

        for x, kv in enumerate(kvc):
            read(kv, z=z, freq=freq, img_array=data, offset=i*chunk_size)

        s = i*chunk_size
        e = s + chunk_size
        write_future = dataset[s:e].write(data)
        writes.append(write_future)

    a = await asyncio.gather(*writes)
    print("done")


def amain(var):
    asyncio.run(main(var))

if __name__ == "__main__":
    app.run(amain)
