import concurrent
import os
from datetime import datetime

import imageio as iio
import numpy as np
import pandas as pd
from absl import flags, app
from absl import logging
from hemera import path_translator as T

from eumetsat import IMG_LAYERS

FLAGS = flags.FLAGS

flags.DEFINE_string("min_date", default="2019-01-01 00:00", help="Min date for image files")
flags.DEFINE_string("max_date", default="2020-12-01 00:00", help="Max date for image files")
flags.DEFINE_integer("freq", default=3600, help="Image update frequency in seconds")


def main(args):
    min_date = datetime.fromisoformat(FLAGS.min_date)
    max_date = datetime.fromisoformat(FLAGS.max_date)

    img_base_path = os.path.join(T.get_path("data"), "EUMETSAT/UK-EXT")

    freq = FLAGS.freq
    freq_min = freq // 60

    date_rage = pd.date_range(min_date, max_date, freq=f"{freq_min}min")
    date_dict = {int(ts.timestamp()): ts.strftime("/year=%Y/month=%m/day=%d/time=%H_%M") for ts in date_rage}
    ts = len(date_dict)

    z = int(date_rage[0].timestamp())

    out_file = os.path.join(T.get_path("data"), "EUMETSAT/UK-EXT", f"imgs_z={z},f={freq}sec")
    img_arry = np.zeros(dtype=np.ubyte, shape=(ts, 500, 500, 12)) #np.memmap(out_file, dtype=np.ubyte, mode='w+', offset=0, shape=(ts, 500, 500, 12))

    def fill_np(date_dict, array):
        """
        Fill the numpy array
        """
        def _read(kv, img_array, index):
            k, v = kv
            t_path = img_base_path + v

            logging.log_every_n(logging.INFO, f"Reading {v}", 100)

            index = (k - z) // freq
            for i, l in enumerate(IMG_LAYERS):
                path = t_path + f"/format={l}/img.png"
                if not os.path.exists(path):
                    logging.info(f"Reading {v}")
                    return
                with iio.imopen(path, "r") as img_file:
                    img = img_file.read(index=0)[..., 0]
                # imgs[:, :, i] = img
                img_array[index, ..., i] = img
            return

        threads = 32

        dates = list(date_dict.items())
        pool = concurrent.futures.ThreadPoolExecutor(threads)

        futures = {}
        for i, kv in enumerate(dates):
            args = (_read,
                    kv,
                    array,
                    i)
            futures[pool.submit(*args)] = i
        concurrent.futures.wait(futures)

        return

    logging.info("Reading now...")
    fill_np(date_dict, img_arry)
    print("Start Flush")
    # img_arry.flush()
    np.save(out_file, img_arry)
    print("Done Flush")
    i = 0

if __name__ == "__main__":
    app.run(main)
