#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PANDA

DATASETS=("skin-lesion")

MODEL_DATA_DIR="./datasets/dataset_skin"
MODEL_OUTPUT_DIR="./results/result_skin-lesion"

for DATASET_NAME in "${DATASETS[@]}"
do
INPUT_FILE_NAME="inputs/skin_$DATASET_NAME.txt"
OUTPUT_FILE_NAME="result_csvs/panda_skin-lesion.csv"
DATASET_DIR="$(pwd)/../../data/Medical/skin-lesion"
TEST_DATASET_DIR="$(pwd)/../../data/Medical/skin-lesion"

if [ -d "$MODEL_DATA_DIR" ]; then
    rm -r "$MODEL_DATA_DIR"
fi
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/train"

ln -sn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/original/$DATASET_NAME/train/good"
ln -sn "$TEST_DATASET_DIR" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$(pwd)/../../data/template/skin-lesion_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=7 python -u infer.py \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$MODEL_DATA_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --batch_size 32 \
    --output_dir "$MODEL_OUTPUT_DIR"

python merge_csv.py "$(pwd)/../../data/template/skin-lesion_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
