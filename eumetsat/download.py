import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import reduce
from multiprocessing import Process, JoinableQueue

import pandas as pd
import pyresample as pr
import requests
from absl import flags, app, logging
from pyproj.crs import CRS
from satpy import Scene
from zephyrus.utils.standard_logger import build_logger

flags.DEFINE_integer('dl', default=2, help="Number of download procs to run")
flags.DEFINE_integer('ep', default=1, help="Number of extractor procs to run")
flags.DEFINE_string('st', default="2020-01-01", help="Start Date")
flags.DEFINE_string('et', default="2021-01-01", help="End Date")
flags.DEFINE_string("dl_base_path", default="/dev/shm", help="Temp download path, RAM disk for high IOPS is good")
flags.DEFINE_string("ext_base_path", default=".", help="Path to save extracted files")
flags.DEFINE_multi_integer("mins", default=[0], help="Minutes of hour to download, any combination of 0, 15, 30, 45")
flags.DEFINE_integer("chunk_size", default=1024, help="Download chunk size in Kb")

FLAGS = flags.FLAGS


# MSG15-RSS
COLLECTION_ID = 'EO:EUM:DAT:MSG:HRSEVIRI'
IMG_LAYERS = ["HRV", "VIS006", "VIS008", "IR_016", "IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]
# read the file
READER = "seviri_l1b_native"

"""" Extract consts """
# TODO we should save this metadata somewhere as we output to PNG so it can get lost
# UK coords in degrees as per WSG84 [llx, lly, urx, ury]
area_extent = [-12., 48., 5., 61.]
area_id = "UK"
description = "Geographical Coordinate System clipped on UK"
proj_dict = {"proj": "longlat", "ellps": "WGS84", "datum": "WGS84"}
proj_crs = CRS.from_user_input(4326).to_dict()  # Target Projection EPSG:4326 standard lat lon geograpic
output_res = [500, 500]  # Target res in pixels
area_def = pr.geometry.AreaDefinition.from_extent(area_id, proj_crs, output_res, area_extent)


def get_dl_path():
    return os.path.join(FLAGS.dl_base_path, "EUMETSAT/RAW")


def get_data_path():
    return os.path.join(FLAGS.ext_base_path, "EUMETSAT/UK-EXT")


@dataclass
class EumetsatToken:
    def __init__(self, key_path: str = "./eumetsat.key"):
        with open(key_path, "r") as f:
            self._key = json.load(f)
        self._last_load = datetime(2017, 5, 1, 0, 0, 0)

    @property
    def token(self) -> str:
        pass

    @token.getter
    def token(self) -> str:
        if (self._last_load - datetime.utcnow()) < timedelta(minutes=50):
            self._last_load = datetime.utcnow()
            self._token = self._load_token()
        return self._token

    def _load_token(self) -> str:
        token = requests.post("https://api.eumetsat.int/token", data="grant_type=client_credentials",
                              auth=(self._key["username"], self._key["password"]))
        return token.json()["access_token"]


def parse_date(ds: str) -> datetime:
    try:
        date = datetime.strptime(ds, "%Y-%m-%dT%H:%M:%S.%fZ")
    except:
        date = datetime.strptime(ds, "%Y-%m-%dT%H:%M:%SZ")
    return date


def ok_hour(date):
    return date.minute in [30]  # TODO: add this as a flag, with defult [0]


def ft(x):
    ds = x["properties"]["date"].split("/")[0]
    date = parse_date(ds)
    return ok_hour(date)


class Extract(Process):
    def __init__(self, files: JoinableQueue):
        Process.__init__(self)
        self.files = files
        self.logger = build_logger("Extractor")

    def run(self):
        running = True
        while running:
            next_task = self.files.get()
            if next_task is None:
                logging.info('Tasks Complete')
                self.files.task_done()
                break
            path = next_task
            start_time = time.monotonic()
            self.make_pngs(path)
            duration = time.monotonic() - start_time
            logging.info(f"Extract took {duration:3.0f}s")
            self.files.task_done()

    def make_pngs(self, path):
        ret = True
        try:
            self.logger.info(f"Loading {path}")
            scn = Scene(filenames={READER: [path]})
            scn.load(scn.all_dataset_names())  # Load all the data inc HRV
            res = scn.resample(area_def, resampler="bilinear")  # cache_dir='/resample_cache/'
            res.save_datasets(writer="simple_image",
                              filename="{start_time:year=%Y/month=%m/day=%d/time=%H_%M}/format={name}/img.png",
                              format="png", base_dir=get_data_path())

        except Exception as e:
            logging.error(e)
            ret = False
        finally:
            # Delete Nat file, keep disk space free as they are big
            logging.info(f"Removing {path}")
            os.remove(path)
            return ret


class Gen(Process):
    # API base endpoint
    apis_endpoint = "https://api.eumetsat.int/data/search-products/os"

    def __init__(self, task_queue: JoinableQueue, collection_id, start: datetime, end: datetime):
        Process.__init__(self)
        self.task_queue = task_queue
        self.items_per_page = 100
        self.collection_id = collection_id
        self.min_date = start
        self.max_date = end
        logging.info(f"Find gaps in Gen for {self.min_date} to {self.max_date}")
        self.gaps = self.find_gaps
        # logging.info(f"gaps {self.gaps}")

    def run(self):
        try:
            self.loop()
        except Exception as e:
            logging.error(e)
        finally:
            logging.info(f"Added all gaps to queue ")

    def between(self, tr1):
        return tr1[0].to_pydatetime() >= self.min_date and tr1[1].to_pydatetime() <= self.max_date

    def find_gaps(self):
        """
        Find the gaps, i.e missing files, to download yielding a range to download.
        This largest range yielded is a month the smallest a single 15 min window
        """
        misssing = []
        search = []

        _months = pd.date_range(self.min_date, self.max_date, freq="MS")
        _months = zip(_months[:-1], _months[1:])
        months = {ts.strftime("year=%Y/month=%m"): (ts, te) for ts, te in _months}
        for mp, tr in months.items():
            time_root = os.path.join(get_data_path(), mp)
            if os.path.exists(time_root):
                search.append(tr)
            else:
                yield tr
                misssing.append(tr)

        # Days
        _days = [pd.date_range(x[0], x[1], freq="D") for x in search]
        _days = [list(zip(dr[:-1], dr[1:])) for dr in _days]
        _days = reduce(lambda a, b: a + b, _days, [])
        days = {ts.strftime("year=%Y/month=%m/day=%d"): (ts, te) for ts, te in _days}
        search = []
        for mp, tr in days.items():
            time_root = os.path.join(get_data_path(), mp)
            if os.path.exists(time_root):
                search.append(tr)
            else:
                yield tr
                misssing.append(tr)

        # Time and Layer check
        _time_feat = [pd.date_range(x[0], x[1], freq="15min") for x in search]
        _time_feat = [list(zip(dr[:-1], dr[1:])) for dr in _time_feat]
        _time_feat = reduce(lambda a, b: a + b, _time_feat, [])
        time_feat = {ts.strftime(f"year=%Y/month=%m/day=%d/time=%H_%M"): (ts, te) for ts, te in _time_feat if
                     ok_hour(ts)}

        for dp, tr in time_feat.items():
            time_root = os.path.join(get_data_path(), dp)
            for l in IMG_LAYERS:
                time_layer = os.path.join(time_root, f"format={l}/img.png")
                if not os.path.exists(time_layer):
                    yield tr
                    break

    def loop(self):
        for range in self.gaps():
            logging.info(f"{range}")
            batch_uf = self.get_unfiltered_range(*range)
            batch = filter(ft, batch_uf)
            # put files on queue
            for el in batch:
                ds = el["properties"]["date"].split("/")[0]
                date = parse_date(ds)
                logging.info(
                    f"Adding File {date.strftime('year=%Y/month=%m/day=%d/time=%H_%M')} to DL Queue")
                self.task_queue.put(el)

    def get_unfiltered_range(self, start_ts: datetime, end_ts: datetime):
        parameters = {'format': 'json',
                      'pi': self.collection_id,
                      "c": self.items_per_page,
                      "si": 0,
                      'dtstart': start_ts.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                      'dtend': end_ts.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                      }

        feats = []
        total = 1
        while parameters["si"] < total:
            response = requests.get(self.apis_endpoint, parameters)
            # Log Failed responses
            if not response.ok:
                logging.error(f"Request for {parameters} Failed: {response.text}")
                break
            found_data_sets = response.json()
            feats.extend(found_data_sets["features"])
            total = found_data_sets['properties']['totalResults']
            parameters["si"] += self.items_per_page
        return feats


class Downloader(Process):
    def __init__(self, task_queue: JoinableQueue, file_queue: JoinableQueue, t: EumetsatToken):
        Process.__init__(self)
        self.task_queue = task_queue
        self.file_queue = file_queue
        self.t = t

    def run(self):
        running = True
        while running:
            file_path = None
            next_task = None
            try:
                next_task = self.task_queue.get()
                if next_task is None:
                    logging.info('Tasks Complete')
                    running = False
                    break
                start_time = time.monotonic()
                file_path = self._run(next_task)
                duration = time.monotonic() - start_time
                logging.info(f"Download took {duration:3.0f}s")
            except Exception as e:
                logging.error(f"Error on {next_task}")
                logging.error(e)
            finally:
                self.task_queue.task_done()
                if file_path:
                    self.file_queue.put(file_path)

    def _run(self, next_task) -> str:
        date = parse_date(next_task["properties"]["date"].split("/")[0])
        sip_ents = next_task['properties']['links']['sip-entries']
        nat_file = filter(lambda x: x['mediaType'] == 'application/octet-stream', sip_ents)
        dl_url = list(nat_file)[0]['href']
        folder = os.path.join(COLLECTION_ID.replace(":", "_"), date.strftime("year=%Y/month=%m/day=%d/time=%H_%M"))
        file_path = self.download(dl_url, folder)
        return file_path

    def download(self, url, base) -> str:
        res = requests.get(url, {"access_token": self.t.token}, stream=True)
        filename = re.findall("\"(.*?)\"", res.headers['Content-Disposition'])[0]
        dir = os.path.join(get_dl_path(), base)
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, filename)
        logging.info(f"{url} -> {path}")
        with open(path, 'wb') as f:
            for c in res.iter_content(chunk_size=int(1024 * FLAGS.chunk_size)):
                f.write(c)
        return path



def main(argv):
    DL_PROCS = FLAGS.dl
    EX_PROCS = FLAGS.ep

    # Define our start and end dates
    start_date = datetime.strptime(FLAGS.st, "%Y-%m-%d")
    end_date = datetime.strptime(FLAGS.et, "%Y-%m-%d")

    # Queue for interprocess communication
    url_q = JoinableQueue(DL_PROCS + int(DL_PROCS * 0.20))
    file_q = JoinableQueue(EX_PROCS + int(EX_PROCS * 0.20) + 2)

    # TODO: use a threadpool and aync maps for this, why are we managing it ourselvs?
    logging.info("Starting Processes")
    # Start Downloader processes
    ex = [Extract(file_q) for _ in range(EX_PROCS)]
    [e.start() for e in ex]
    [Downloader(url_q, file_q, EumetsatToken()).start() for _ in range(DL_PROCS)]

    logging.info(f"Starting Gen for {start_date} to {end_date}")
    # Start and join generator
    g = Gen(url_q, COLLECTION_ID, start_date, end_date)
    g.start()
    g.join()  # wait for generator to add all urls

    # Join queue and put `final` message to end downloader
    logging.info(f"Added Termination Singles to DOWNLOAD queue")
    [url_q.put(None) for _ in range(DL_PROCS)]
    url_q.join()  # wait for files to download
    logging.info("URL Queue Done")

    logging.info(f"Added Termination Singles to Extract queue")
    [file_q.put(None) for _ in range(EX_PROCS)]
    [e.join() for e in ex]
    logging.info("All Procs Done")


if __name__ == "__main__":
    # Parse Args
    app.run(main)
