import concurrent
import os
import os.path
from datetime import datetime

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
from eumetsat import IMG_LAYERS
from absl import flags, app, logging
from hemera.path_translator import get_path

flags.DEFINE_string("min_date", default="2018-01-01 00:00", help="Min date for image files")
flags.DEFINE_string("max_date", default="2018-01-03 00:00", help="Max date for image files")
flags.DEFINE_integer("freq", default=900, help="Numpy data blob")
flags.DEFINE_boolean("save", default=True, help="Save the dataset")

FLAGS = flags.FLAGS


def read(min_date, max_date):
    img_base_path = os.path.join(get_path("data"), "EUMETSAT/UK-EXT")

    freq = FLAGS.freq
    freq_min = freq // 60

    date_rage = pd.date_range(min_date, max_date, freq=f"{freq_min}min")
    date_dict = {int(ts.timestamp()): ts.strftime("/year=%Y/month=%m/day=%d/time=%H_%M") for ts in date_rage}
    ts = len(date_dict)

    z = int(date_rage[0].timestamp())

    img_arry = np.zeros(dtype=np.ubyte, shape=(ts, 500, 500, 12))

    def fill_np(date_dict, array):
        """
        Fill the numpy array
        """
        def _read(kv, img_array, index):
            k, v = kv
            t_path = img_base_path + v
            logging.debug(f"Reading {t_path}")
            index = (k - z) // freq
            for i, l in enumerate(IMG_LAYERS):
                path = t_path + f"/format={l}/img.png"
                if not os.path.exists(path):
                    logging.warning(f"File not found {v}")
                    return
                with iio.imopen(path, "r") as img_file:
                    img = img_file.read(index=0)[..., 0]
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
    return img_arry


def main(args):
    ts_start = datetime.fromisoformat(FLAGS.min_date)
    ts_end = datetime.fromisoformat(FLAGS.max_date)


    def gen():
        freq = FLAGS.freq
        freq_min = freq // 60
        date_rage = pd.date_range(ts_start, ts_end, freq=f"{freq_min}min")

        s0 = date_rage[::(4 * 24 * 30)]
        s1 = s0[1:].append(date_rage[-1:])
        for start, end in zip(s0, s1):
            ts_zero = int(start.timestamp())
            im_data = read(start, end)
            idx = np.arange(len(im_data))  # Note we take off window from len so we can have time steps
            idx = idx
            for x in idx:
                ix = (x * FLAGS.freq) + ts_zero
                exists = True
                if im_data[x, 100:104, 100:104].sum() == 0:
                   exists = False
                yield np.array(ix, dtype=np.int64), np.array(exists, dtype=bool), im_data[x]

            del im_data

    # l = gen()
    # v = list(l)
    shard_size = (60 * 60 * 24 * 30) # Shard by arpox months
    def shard(ts, *x):
        x = (ts - int(ts_start.timestamp())) // shard_size
        return tf.cast(x, tf.int64) # Shard by arpox months


    data = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(), dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.bool),
                                                                 tf.TensorSpec(shape=(500, 500, 12), dtype=tf.uint8)))


    if FLAGS.save:
        path = os.path.join(get_path("data"), f"EUMETSAT/UK-EXT/img_z={ts_start.timestamp()},e={ts_end.timestamp()},f=900,s={shard_size}.gz.tfds")
        data.save(path, compression="GZIP", shard_func=shard)

    # Window Read back in
    telspec = (tf.TensorSpec(shape=(3,), dtype=tf.int64), tf.TensorSpec(shape=(3,), dtype=tf.bool), tf.TensorSpec(shape=(3, 500, 500, 12), dtype=tf.uint8))

    def winder(ds):
        ds = ds.window(3, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda *xs: tf.data.Dataset.zip((*xs,)).batch(3, drop_remainder=True))
        return ds

    def wr(ds):
        return ds.interleave(lambda x: winder(x))

    ds = tf.data.Dataset.load(path, element_spec=telspec, compression="GZIP", reader_func=wr) if FLAGS.save else data
    ds = ds.filter(lambda ts, f, x: tf.reduce_all(f))
    # ds = ds.window(3, shift=1, drop_remainder=True)
    # ds = ds.flat_map(lambda x,y : tf.data.Dataset.zip((x,y)).batch(3, drop_remainder=True))
    res = list(ds.take(193))

    return

if __name__ == "__main__":
    app.run(main)
