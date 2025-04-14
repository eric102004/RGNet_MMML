#!/usr/bin/env bash


source activate rgnet

rgnet/scripts/inference_mad.sh >log/inference_mad6.log 2>&1
