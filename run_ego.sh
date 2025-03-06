#!/usr/bin/env bash


source activate rgnet

mode=finetune

if [ $mode == "finetune" ]; then
    rgnet/scripts/finetune_ego4d.sh >log/finetune_ego4d.log 2>&1
elif [ $mode == "inference" ]; then
    rgnet/scripts/inference_ego4d.sh >log/inference_ego4d.log 2>&1
fi

