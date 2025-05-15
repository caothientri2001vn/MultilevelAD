#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

DATASETS=("diabetic")

INPUT_FILE_NAME="test_input_files_diabetic.txt"
MODEL_DATA_DIR="./dataset_diabetic"
MODEL_OUTPUT_DIR="./result_diabetic"

for DATASET_NAME in "${DATASETS[@]}"
do
OUTPUT_FILE_NAME="result_csvs/pni_diabetic-retinopathy.csv"
DATASET_DIR="$(pwd)/../../data/Medical/diabetic-retinopathy"
TEST_DATASET_DIR="$(pwd)/../../data/Medical/diabetic-retinopathy"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/original"


mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/train/"

ln -sn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/original/$DATASET_NAME/train/good"

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good/$(basename "$file")"
done

find "$DATASET_DIR/level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad/$(basename "$file")"
done


ln -sn "$TEST_DATASET_DIR" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$(pwd)/../../data/template/diabetic-retinopathy_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=2 python -u infer.py \
    --category $DATASET_NAME \
    --seed 23 \
    --dataset_path "$MODEL_DATA_DIR/original" \
    --project_root_path "$MODEL_OUTPUT_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --test_dir "$MODEL_DATA_DIR/test"

python merge_csv.py "$(pwd)/../../data/template/diabetic-retinopathy_template.csv" "$OUTPUT_FILE_NAME" "merged_${OUTPUT_FILE_NAME}"
done
