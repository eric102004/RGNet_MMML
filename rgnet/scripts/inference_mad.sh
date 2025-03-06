#!/usr/bin/env bash

ckpt_path=mad_checkpoint/mad_trained_e0032.ckpt
eval_split_name=test
device_id=0
eval_id=test
#eval_path=data/mad_test.jsonl
eval_path=data/mad_data_for_cone/data/mad_data/test.jsonl
echo ${eval_path}
#CUDA_VISIBLE_DEVICES=${device_id} 
PYTHONPATH=$PYTHONPATH:. python rgnet/inference.py \
    --resume ${ckpt_path} \
    --eval_split_name ${eval_split_name} \
    --eval_path ${eval_path} \
    --eval_id ${eval_id} \
    --num_workers 8 \
    --ret_eval \
    --topk_window 30 \
    --nms_thd 0.5 \
    --topk 10 \
    --topk_span 5 \
    --max_after_nms 100 \
    ${@:2}
