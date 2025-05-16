#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

DATASETS=("bichon_frise" "chinese_rural_dog" "golden_retriever" "labrador_retriever" "teddy")

MODEL_DATA_DIR="./dataset_multidog"
MODEL_OUTPUT_DIR="./result_multidog"

for DATASET_NAME in "${DATASETS[@]}"
do
DATASET_DIR="$(pwd)/../../data/OneClassNovelty/multidog"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test/good"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test/bad"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/train/"

ln -sn "$DATASET_DIR/level_0_train/$DATASET_NAME" "$MODEL_DATA_DIR/$DATASET_NAME/train/good"

find "$DATASET_DIR/level_0_test/$DATASET_NAME" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/$DATASET_NAME/test/good/$(basename "$file")"
done

find "$DATASET_DIR/level_1/other_dogs" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/$DATASET_NAME/test/bad/$(basename "$file")"
done

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
