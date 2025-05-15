#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "toothbrush" "transistor" "zipper")

MODEL_DATA_DIR="./dataset_mvtec"
MODEL_OUTPUT_DIR="./result_mvtec"

for DATASET_NAME in "${DATASETS[@]}"
do
DATASET_DIR="$(pwd)/../../data/Industry/mvtec/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR"
ln -sn "$DATASET_DIR" "$MODEL_DATA_DIR/$DATASET_NAME"

echo "Training on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=6 python -u train_coreset_distribution.py \
    --category $DATASET_NAME \
    --seed 23 \
    --train_coreset \
    --train_nb_dist \
    --train_coor_dist \
    --dataset_path "$MODEL_DATA_DIR" \
    --project_root_path "$MODEL_OUTPUT_DIR"

done
