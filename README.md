# Eumetsat-downloader

A small python tall to download and crop images from [eumetstat](https://www.eumetsat.int/)


## Set Up
The easiest way is to clone the repo and install it. 
Hopefully if I have the config set up it should install all the deps
```shell
git clone 
pip install -e .
```
Then to run some models, assuming the data dict is set up
```shell
python ./eumetsat/download.py --dl 4 --ep 16 --st "2015-01-01" --et "2021-01-01"
```

