#!/usr/bin/env bash
set -e
exec > output_infer_visa.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

# DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")
DATASETS=("pcb3" "pipe_fryum")
# DATASETS=("bottle")

INPUT_FILE_NAME="test_input_files_visa.txt"
MODEL_DATA_DIR="./dataset_visa"
MODEL_OUTPUT_DIR="./result_visa"

for DATASET_NAME in "${DATASETS[@]}"
do
OUTPUT_FILE_NAME="an_scores_${DATASET_NAME}.csv"
DATASET_DIR="$(pwd)/../../data/area-based/VisA_reorganized/$DATASET_NAME"
TEST_DATASET_DIR="$(pwd)/../../data/area-based/VisA_reorganized/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/original"


mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/ground_truth"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/test"
mkdir -p "$MODEL_DATA_DIR/original/$DATASET_NAME/train"
ln -sn "$DATASET_DIR/ground_truth/bad" "$MODEL_DATA_DIR/original/$DATASET_NAME/ground_truth/bad"
ln -sn "$DATASET_DIR/test/bad" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/bad"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/original/$DATASET_NAME/test/good"
ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/original/$DATASET_NAME/train/good"


ln -sn "$TEST_DATASET_DIR/test/" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=3 python -u infer.py \
    --category $DATASET_NAME \
    --seed 23 \
    --dataset_path "$MODEL_DATA_DIR/original" \
    --project_root_path "$MODEL_OUTPUT_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --test_dir "$MODEL_DATA_DIR/test"

python merge_csv.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_pni_visa_${DATASET_NAME}.csv"
done
