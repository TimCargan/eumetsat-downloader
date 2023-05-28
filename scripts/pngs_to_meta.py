import serde
from datetime import datetime
from pathlib import Path

import pandas as pd
import serde
import serde.json
from absl import flags, app
from absl import logging
from rich.progress import Progress

from eumetsat import IMG_LAYERS
from eumetsat.datasets.utils import Metadata
from hemera import path_translator

flags.DEFINE_integer("freq", default=900, help="Expected freq (in seconds) to scan for images")

FLAGS = flags.FLAGS



def datepath_ok(path: Path) -> bool:
    for i, l in enumerate(IMG_LAYERS):
        ipath = path / f"format={l}/img.png"
        if not ipath.exists():
            return False
    return True

def min_max(path:Path, glob_pattern:str) -> (int, int):
    items = path.glob(glob_pattern)
    items = sorted(items)
    min_value = int(str(items[0]).split("=")[-1])
    max_value = int(str(items[-1]).split("=")[-1])
    return min_value, max_value

def main(args):
    img_base_path = Path(path_translator.get_path("data")) / "EUMETSAT/UK-EXT"
    out_file = img_base_path / "png_metadata.json"

    freq = FLAGS.freq
    freq_min = freq // 60

    min_year, max_year = min_max(img_base_path, "year=*")
    min_year_month = min_max(img_base_path / f"year={min_year}", "month=*")[0]
    max_year_month = min_max(img_base_path / f"year={max_year}", "month=*")[1]

    search_start = datetime(year=min_year, month=min_year_month, day=1)
    search_end = datetime(year=max_year + max_year_month//12, month=(max_year_month + 1)%12 , day=1)  # Funcky maths to go to the start of the next month
    pd.date_range(search_start, search_end)
    search_range = pd.date_range(search_start, search_end, freq=f"{freq_min}min", inclusive="left")

    logging.info("Scanning now... from %d to %d", min_year, max_year)
    missing = []
    with Progress() as p:
        task = p.add_task("[green]Scanning...", total=len(search_range))
        for d in search_range:
            p.update(task, advance=1)
            test_path = img_base_path / d.strftime("year=%Y/month=%m/day=%d/time=%H_%M")
            ok = datepath_ok(test_path)
            if not ok:
                missing.append(d)

    logging.info("missing %d of %d", len(missing), len(search_range))
    logging.info("Writing meta")
    metadata = Metadata(
        metadata_created=datetime.now(),
        first_example_date=search_start,
        last_example_date=search_end,
        example_count=len(search_range) - len(missing),
        missing=set(missing),
        freq_seconds=freq,
        data_source=Path("external:EUMETSAT"),
        data_location=img_base_path
    )

    metadata_str = serde.json.to_json(metadata)
    with out_file.open("w") as f:
        f.write(metadata_str)

    logging.info("Done")

if __name__ == "__main__":
    app.run(main)
