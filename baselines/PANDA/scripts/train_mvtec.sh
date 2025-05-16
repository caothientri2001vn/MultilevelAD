#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PANDA

DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")

MODEL_DATA_DIR="./datasets/dataset_mvtec"
MODEL_OUTPUT_DIR="./results/result_mvtec"

for DATASET_NAME in "${DATASETS[@]}"
do
DATASET_DIR="$(pwd)/../../data/Industry/mvtec/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR"
ln -sn "$DATASET_DIR" "$MODEL_DATA_DIR/$DATASET_NAME"

echo "Training on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=7 python -u panda.py \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$MODEL_DATA_DIR/$DATASET_NAME" \
    --epochs 30 \
    --batch_size 32 \
    --output_dir "$MODEL_OUTPUT_DIR"

done
