#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;


# make sure to call this script at `multilingual-keyphrase-generation/generator`

HOME_DIR=`realpath ../`
DATA_DIR="${HOME_DIR}/data/taboola/mbart/"


MODEL=facebook/mbart-large-50
# MODEL=lincoln/mbart-mlsum-automatic-summarization
MODEL_SHORT=mbart-large-50
# MODEL_SHORT=mbart-large-en-mkp
# MODEL_SHORT=mbart-mlsum-lincoln


function run_ddp_train() {
export CUDA_VISIBLE_DEVICES="3"
export OMP_NUM_THREADS=1
N_GPU=1
MODEL=$1
DATASET=$2

OUTPUT_DIR="${HOME_DIR}/models/"
mkdir -p ${OUTPUT_DIR}

EP=20   # 10 for small datasets
LR=3e-5
BATCH_SIZE_PER_GPU=8
GRAD_ACCUMULATION_STEPS=4   # total effective batch 32

TRAIN="${DATA_DIR}/${DATASET}/mix.train.json"
DEV="${DATA_DIR}/${DATASET}/mix.dev.json"
TEST="${DATA_DIR}/${DATASET}/mix.test.json"

MODEL_DIR=${OUTPUT_DIR}/taboola/$(date +'%Y%m%d-%H%M')_${MODEL_SHORT}_${DATASET}_checkpoints_ddp_${N_GPU}gpu_e${EP}_lr${LR}_pergpubsz${BATCH_SIZE_PER_GPU}x${GRAD_ACCUMULATION_STEPS}/
python -m torch.distributed.launch  --nproc_per_node ${N_GPU}  --master_port=4684 train_multilingual_kpgen.py \
       --langs "af-af_ZA;de-de_DE;es-es_XX;fr-fr_XX;he-he_IL;hi-hi_IN;id-id_ID;it-it_IT;ja-ja_XX;ko-ko_KR;nl-nl_XX;pt-pt_XX;ro-ro_RO;sv-sv_SE;th-th_TH" \
       --predict_with_generate \
       --output_dir=${MODEL_DIR} \
       --overwrite_output_dir \
       --do_train \
       --do_eval \
       --do_predict \
       --evaluation_strategy=epoch \
       --save_strategy=epoch \
       --save_total_limit=1 \
       --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
       --per_device_eval_batch_size=${BATCH_SIZE_PER_GPU} \
       --gradient_accumulation_steps=${GRAD_ACCUMULATION_STEPS} \
       --learning_rate=${LR} \
       --weight_decay=0.01 \
       --num_train_epochs=${EP} \
       --warmup_steps=10 \
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



DATASET=mixed_all_15_langs
# for DATASET in `ls /local1/diwu/multilingual-keyphrase-generation/data/taboola/mbart/ | grep _only`; do
    run_ddp_train ${MODEL} ${DATASET}
    rm -r ${MODEL_DIR}/checkpoint-*
# done
