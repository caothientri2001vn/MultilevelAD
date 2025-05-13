#!/usr/bin/env bash
set -e
exec > logs/infer_skin_$1.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

DATASETS=("skin")

INPUT_FILE_NAME="input_list/skin_${1}.txt"
MODEL_DATA_DIR="./dataset_skin_${1}"
MODEL_OUTPUT_DIR="./result_skin"

for DATASET_NAME in "${DATASETS[@]}"
do
OUTPUT_FILE_NAME="csv_corrupted/pni_skin-lesion_corrupted_$1.csv"
DATASET_DIR="$(pwd)/../../data/severity-based/skin-lesion"
TEST_DATASET_DIR="$(pwd)/../../data/corrupted/$1/skin-lesion"

if [[ -d "$MODEL_DATA_DIR" ]]; then
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

find "$DATASET_DIR/NV_level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sfn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad/$(basename "$file")"
done


ln -sn "$TEST_DATASET_DIR" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$TEST_DATASET_DIR/skin-lesion_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=$2 python -u infer.py \
    --category $DATASET_NAME \
    --seed 23 \
    --dataset_path "$MODEL_DATA_DIR/original" \
    --project_root_path "$MODEL_OUTPUT_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --test_dir "$MODEL_DATA_DIR/test"

python merge_csv.py "$TEST_DATASET_DIR/skin-lesion_template.csv" "$OUTPUT_FILE_NAME" "merged_${OUTPUT_FILE_NAME}"
done
