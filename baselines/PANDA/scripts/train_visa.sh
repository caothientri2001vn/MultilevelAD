#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PANDA

DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")

MODEL_DATA_DIR="./datasets/dataset_visa"
MODEL_OUTPUT_DIR="./results/result_visa"

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
CUDA_VISIBLE_DEVICES=6 python -u panda.py \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$MODEL_DATA_DIR/$DATASET_NAME" \
    --epochs 30 \
    --batch_size 32 \
    --output_dir "$MODEL_OUTPUT_DIR"

done
