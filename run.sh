#!/usr/bin/env bash


source activate rgnet

rgnet/scripts/inference_mad.sh >log/inference.log 2>&1
