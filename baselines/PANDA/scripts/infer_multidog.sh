#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PANDA

DATASETS=("bichon_frise" "chinese_rural_dog" "golden_retriever" "labrador_retriever" "teddy")

MODEL_DATA_DIR="./datasets/dataset_multidog"
MODEL_OUTPUT_DIR="./results/result_multidog"

for DATASET_NAME in "${DATASETS[@]}"
do
INPUT_FILE_NAME="inputs/visa_$DATASET_NAME.txt"
OUTPUT_FILE_NAME="result_csvs/panda_example_${DATASET_NAME}.csv"
DATASET_DIR="$(pwd)/../../data/OneClassNovelty/multidog"
TEST_DATASET_DIR="$(pwd)/../../data/OneClassNovelty/multidog"

if [ -d "$MODEL_DATA_DIR" ]; then
    rm -r "$MODEL_DATA_DIR"
fi
mkdir -p "$MODEL_DATA_DIR/original"


mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/train/"

ln -sn "$DATASET_DIR/level_0_train/$DATASET_NAME" "$MODEL_DATA_DIR/original/$DATASET_NAME/train/good"

find "$DATASET_DIR/level_0_test/$DATASET_NAME" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good/$(basename "$file")"
done

find "$DATASET_DIR/level_1/other_dogs" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad/$(basename "$file")"
done


ln -sn "$TEST_DATASET_DIR" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$(pwd)/../../data/template/multidog_${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=7 python -u infer.py \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$MODEL_DATA_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --batch_size 32 \
    --output_dir "$MODEL_OUTPUT_DIR"

python merge_csv.py "$(pwd)/../../data/template/multidog_${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
