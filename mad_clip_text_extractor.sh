#!/usr/bin/env bash

source activate rgnet

python mad_clip_text_extractor.py >log/mad_clip_text_extractor.log 2>&1
