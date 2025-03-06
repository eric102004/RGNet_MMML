ckpt_path=ego4d_checkpoint/ego_finetune_best.ckpt
eval_split_name=val
eval_path=data/ego4d_nlq_data_for_cone/data/ego4d_data/val.jsonl
eval_id=val
device_id=0
echo ${eval_path}
CUDA_VISIBLE_DEVICES=${device_id} PYTHONPATH=$PYTHONPATH:. python rgnet/inference.py \
    --resume ${ckpt_path} \
    --eval_split_name ${eval_split_name} \
    --eval_path ${eval_path} \
    --eval_id ${eval_id} \
    --ret_eval \
    --topk_window 20 \
    --nms_thd 0.5 \
    --topk 10 \
    --topk_span 5 \
    --max_after_nms 100
    ${@:2}
