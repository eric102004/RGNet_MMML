#!/usr/bin/env bash

source activate rgnet

python feature_extraction/misc/convert_h5_to_lmdb.py >log/convert_h5_to_lmdb.log 2>&1
