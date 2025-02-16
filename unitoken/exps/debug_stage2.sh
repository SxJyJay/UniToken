#!/bin/bash

lr=1e-5
min_lr=0
vit_lr_scale=1
wd=0.1
dropout=0.05
z_loss_weight=1e-5
macro_bs=2

# Single node launch
NNODES=1
GPUS_PER_NODE=2
MASTER_ADDR='localhost'
MASTER_PORT=6001
NODE_RANK=0

# For multinode launch, set above variable from environment

distributed_args="\
    --nnodes $NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
"

accum_iter=1
micro_bs=$(($macro_bs/$NNODES/$GPUS_PER_NODE/$accum_iter))
ep=1

init_from=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/jiaoyang/jiaoyang/UniToken/unitoken/output/DEBUG-UniToken-base-stage1/epoch0-iter0
data_config=configs/data/config_phase1_allava710k_llavadetail559k_sharecap1246k.yaml
trainable_params=vit,adapter,model,lm_head

exp_name=DEBUG-UniToken-base-stage2

# Local path
ws_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/jiaoyang/jiaoyang
proj_dir=${ws_dir}/UniToken/unitoken
cd ${proj_dir}

output_root=${proj_dir}/output
output_dir=${output_root}/"$exp_name"
mkdir -p ${output_dir}

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
torchrun ${distributed_args} finetune_solver_base.py \
--init_from ${init_from} \
--trainable_params ${trainable_params} \
--vit_lr_scale ${vit_lr_scale} \
--batch_size ${micro_bs} \
--accum_iter ${accum_iter} \
--epochs ${ep} \
--warmup_epochs 0.01 \
--lr ${lr} \
--min_lr ${min_lr} \
--wd ${wd} \
--clip_grad 4 \
--data_config $data_config \
--cache_ann_on_disk \
--num_workers 8 \
--output_dir ${output_dir} \
--save_iteration_interval 1000 \
--checkpointing \
--max_seq_len 4096 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
--stage II \
2>&1 | tee -a ${output_dir}/output_node${NODE_RANK}.log

echo "exp name: $exp_name"
