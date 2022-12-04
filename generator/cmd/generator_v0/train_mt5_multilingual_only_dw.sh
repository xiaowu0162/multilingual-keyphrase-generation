#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;


function run_ddp_train() {
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=1
N_GPU=1  # 4
OUTPUT_DIR=$1
MODEL=$2
DATASET=$3

EP=30
LR=5e-5   # 1e-4
BATCH_SIZE_PER_GPU=1
GRAD_ACCUMULATION_STEPS=16   # to match the original bsz setting use 48


DATA="/home/diwu/kpgen/multilingual-keyphrase-generation/data/e-commerce"
TRAIN="${DATA}/${DATASET}/mix.train.json"
DEV="${DATA}/${DATASET}/mix.dev.json"
TEST="${DATA}/${DATASET}/mix.test.json"

#deepspeed train_multilingual_kpgen_mt5.py
python -m torch.distributed.launch --nproc_per_node ${N_GPU} --master_port=4684 \
       train_multilingual_kpgen_mt5.py \
       --predict_with_generate \
       --output_dir=${OUTPUT_DIR}/$(date +'%Y%m%d-%H%M')_${DATASET}_checkpoints_ddp_${N_GPU}gpu_e${EP}_lr${LR}_pergpubsz${BATCH_SIZE_PER_GPU}x${GRAD_ACCUMULATION_STEPS} \
       --overwrite_output_dir \
       --do_train \
       --do_eval \
       --do_predict \
       --evaluation_strategy=epoch \
       --save_strategy=epoch \
       --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
       --per_device_eval_batch_size=${BATCH_SIZE_PER_GPU} \
       --gradient_accumulation_steps=${GRAD_ACCUMULATION_STEPS} \
       --learning_rate=${LR} \
       --weight_decay=0.01 \
       --num_train_epochs=${EP} \
       --warmup_steps=100 \
       --logging_steps=20 \
       --seed=95 \
       --load_best_model_at_end \
       --metric_for_best_model=eval_f1 \
       --greater_is_better=True \
       --model_name_or_path=${MODEL} \
       --source_lang=ar_AR \
       --target_lang=ar_AR \
       --train_file=${TRAIN} \
       --validation_file=${DEV} \
       --test_file=${TEST} \
       --overwrite_cache \
       --num_beams=5

# --fp16
# --deepspeed "deepspeed_configs/normal_fp16_gpuonly.json" 
# --fp16 true 
# --half_precision_backend "auto"

}



DATASET=mix_de_es_fr_it
MODEL=google/mt5-small   # facebook/mbart-large-cc25

OUTPUT_DIR="/local/diwu/multilingual_kpgen_experiments/mt5-small/"
mkdir -p $OUTPUT_DIR

run_ddp_train ${OUTPUT_DIR} ${MODEL} ${DATASET}

