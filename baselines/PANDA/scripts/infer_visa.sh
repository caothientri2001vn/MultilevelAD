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
INPUT_FILE_NAME="inputs/visa_$DATASET_NAME.txt"
OUTPUT_FILE_NAME="result_csvs/panda_visa_${DATASET_NAME}.csv"
DATASET_DIR="$(pwd)/../../data/Industry/visa/$DATASET_NAME"
TEST_DATASET_DIR="$(pwd)/../../data/Industry/visa/$DATASET_NAME"

if [ -d "$MODEL_DATA_DIR" ]; then
    rm -r "$MODEL_DATA_DIR"
fi
mkdir -p "$MODEL_DATA_DIR/original"

mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/ground_truth"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/train"
ln -sn "$DATASET_DIR/ground_truth/bad" "$MODEL_DATA_DIR/original/$DATASET_NAME/ground_truth/bad"
ln -sn "$DATASET_DIR/test/bad" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good"
ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/original/$DATASET_NAME/train/good"

ln -sn "$TEST_DATASET_DIR/test/" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$(pwd)/../../data/template/visa_${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=7 python -u infer.py \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$MODEL_DATA_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --batch_size 32 \
    --output_dir "$MODEL_OUTPUT_DIR"

python merge_csv.py "$(pwd)/../../data/template/visa_${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
