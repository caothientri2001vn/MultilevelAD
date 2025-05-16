#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PANDA

DATASETS=("diabetic")

MODEL_DATA_DIR="./datasets/dataset_diabetic"
MODEL_OUTPUT_DIR="./results/result_diabetic"

for DATASET_NAME in "${DATASETS[@]}"
do
DATASET_DIR="$(pwd)/../../data/Medical/diabetic-retinopathy"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test/good"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test/bad"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/train"

ln -sn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/$DATASET_NAME/train/good"

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/$DATASET_NAME/test/good/$(basename "$file")"
done

find "$DATASET_DIR/level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/$DATASET_NAME/test/bad/$(basename "$file")"
done

echo "Training on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=7 python -u panda.py \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$MODEL_DATA_DIR/$DATASET_NAME" \
    --epochs 30 \
    --batch_size 32 \
    --output_dir "$MODEL_OUTPUT_DIR"

done
