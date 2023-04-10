#!/bin/bash

#SBATCH -J DL
#SBATCH -q cpu -p cpu
#SBATCH -c 32 --mem=136G

python ./download.py --dl 4 --ep 16 --st "2015-01-01" --et "2021-01-01"
