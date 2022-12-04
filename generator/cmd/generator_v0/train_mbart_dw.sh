#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;


HOME_DIR=`realpath ../../..`
DATA_DIR="${HOME_DIR}/data/e-commerce"


MODEL=facebook/mbart-large-cc25
MODEL_SHORT=mbart-large


function run_ddp_train() {
export CUDA_VISIBLE_DEVICES="4"
export OMP_NUM_THREADS=1
N_GPU=1
OUTPUT_DIR=$1
MODEL=$2
DATASET=$3

OUTPUT_DIR="${HOME_dir}/models/$(date +'%Y%m%d-%H%M')_${MODEL_SHORT}_${DATASET}_checkpoints"
mkdir -p $OUTPUT_DIR

EP=10
LR=1e-4
BATCH_SIZE_PER_GPU=8
GRAD_ACCUMULATION_STEPS=4

TRAIN="${DATA_DIR}/${DATASET}/mix.train.json"
DEV="${DATA_DIR}/${DATASET}/mix.dev.json"
TEST="${DATA_DIR}/${DATASET}/mix.test.json"

python -m torch.distributed.launch  --nproc_per_node ${N_GPU}  --master_port=4684 train_multilingual_kpgen.py \
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
       --logging_steps=200 \
       --seed=95 \
       --fp16 \
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
}


# DATASET=mix_de_es_fr_it
for DATASET in mix_de_es_fr_it de_only es_only fr_only it_only; do
    run_ddp_train ${OUTPUT_DIR} ${MODEL} ${DATASET}
done
