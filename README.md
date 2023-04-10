# EUMETSAT 
This is a collection of library and scripts to download and deal with data from [eumetstat](https://www.eumetsat.int/). 
Specifically data covering the UK downloaded from the high rate SEVIRI imager from metosat 2nd gen (EO:EUM:DAT:MSG:HRSEVIRI).


## Eumetsat-downloader

A python script to download and crop images from [eumetstat](https://www.eumetsat.int/). 
It downloaded the data and extracts a 500 x 500px png file for each of the 12 layers.
It might not be the best code, but it works (most of the time).

### Set Up
The easiest way is to clone the repo and install it. 
Hopefully if I have the config set up it should install all the deps
```shell
git clone https://github.com/TimCargan/eumetsat.git
cd eumetsat
pip install -e .
```
Go to eumetsat website and get your key, put it in `/scripts/eumetsat.key`.
Then to run the download script, assuming the data dict is set up use:
```shell
python ./scripts/download.py --dl 2 --ep 6 --st "2015-01-01" --et "2021-01-01" --mins 0 --mins 30
```

## Data Processing
In addition to downloading the data there are some scripts to shape the raw pngs into a more ml friendly formats:
- tf.data
- tensorstore
- numpy memmap file
