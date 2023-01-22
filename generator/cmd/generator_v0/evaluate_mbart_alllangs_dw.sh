#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;


# make sure to call this script at `multilingual-keyphrase-generation/generator`

HOME_DIR=`realpath ../`
DATA_DIR="${HOME_DIR}/data/e-commerce"


function run_eval() {
export CUDA_VISIBLE_DEVICES="5"
export OMP_NUM_THREADS=1
MODEL=$1
DATASET=$2

EP=3   # 10 for small datasets
LR=1e-4
BATCH_SIZE_PER_GPU=32
GRAD_ACCUMULATION_STEPS=1   # total effective batch 32

TRAIN="${DATA_DIR}/${DATASET}/mix.train.json"
DEV="${DATA_DIR}/${DATASET}/mix.dev.json"
TEST="${DATA_DIR}/${DATASET}/mix.test.json"

python train_multilingual_kpgen.py \
       --predict_with_generate \
       --output_dir=${OUTPUT_DIR} \
       --overwrite_output_dir \
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
       --tokenizer_name="facebook/mbart-large-cc25" \
       --source_lang=ar_AR \
       --target_lang=ar_AR \
       --test_file=${TEST} \
       --overwrite_cache \
       --num_beams=5
}


DATASET=mix_de_es_fr_it
OUTPUT_DIR=$1
run_eval ${OUTPUT_DIR} ${DATASET}
