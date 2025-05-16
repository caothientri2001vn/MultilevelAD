#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..

# exec > logs/output_infer_diabetic.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PANDA

DATASETS=("diabetic")

MODEL_DATA_DIR="./datasets/dataset_diabetic"
MODEL_OUTPUT_DIR="./results/result_diabetic"

for DATASET_NAME in "${DATASETS[@]}"
do
INPUT_FILE_NAME="inputs/diabetic.txt"
OUTPUT_FILE_NAME="csvs/panda_diabetic.csv"
DATASET_DIR="$(pwd)/../../data/severity-based/diabetic-retinopathy"
TEST_DATASET_DIR="$(pwd)/../../data/severity-based/diabetic-retinopathy"

if [ -d "$MODEL_DATA_DIR" ]; then
    rm -r "$MODEL_DATA_DIR"
fi
mkdir -p "$MODEL_DATA_DIR/original"


mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/train/good"

# Process each file in the subdirectory
find "$DATASET_DIR/level_0_train" -maxdepth 1 -type f | sort | head -n 1000 | while IFS= read -r file; do
  ln -sfn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/train/good/$(basename "$file")"
done

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sfn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good/$(basename "$file")"
done

find "$DATASET_DIR/level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sfn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad/$(basename "$file")"
done


ln -sn "$TEST_DATASET_DIR" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=7 python -u infer.py \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$MODEL_DATA_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --batch_size 32 \
    --output_dir "$MODEL_OUTPUT_DIR"

python merge_csv.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done