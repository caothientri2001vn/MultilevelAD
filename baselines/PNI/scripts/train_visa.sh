#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")

MODEL_DATA_DIR="./dataset_visa"
MODEL_OUTPUT_DIR="./result_visa"

for DATASET_NAME in "${DATASETS[@]}"
do
DATASET_DIR="$(pwd)/../../data/Industry/visa/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/ground_truth"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/train"
ln -sn "$DATASET_DIR/ground_truth/bad" "$MODEL_DATA_DIR/$DATASET_NAME/ground_truth/bad"
ln -sn "$DATASET_DIR/test/bad" "$MODEL_DATA_DIR/$DATASET_NAME/test/bad"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/$DATASET_NAME/test/good"
ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/$DATASET_NAME/train/good"

echo "Training on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=3 python -u train_coreset_distribution.py \
    --category $DATASET_NAME \
    --seed 23 \
    --train_coreset \
    --train_nb_dist \
    --train_coor_dist \
    --dataset_path "$MODEL_DATA_DIR" \
    --project_root_path "$MODEL_OUTPUT_DIR"

done
