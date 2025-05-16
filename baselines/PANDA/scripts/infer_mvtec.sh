#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..

# exec > logs/output_infer_mvtec.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PANDA

DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")

MODEL_DATA_DIR="./datasets/dataset_mvtec"
MODEL_OUTPUT_DIR="./results/result_mvtec"

for DATASET_NAME in "${DATASETS[@]}"
do
INPUT_FILE_NAME="inputs/mvtec_$DATASET_NAME.txt"
OUTPUT_FILE_NAME="csvs/panda_mvtec_${DATASET_NAME}.csv"
DATASET_DIR="$(pwd)/../../data/area-based/mvtec_order/$DATASET_NAME"
TEST_DATASET_DIR="$(pwd)/../../data/area-based/mvtec_order/$DATASET_NAME"

if [ -d "$MODEL_DATA_DIR" ]; then
    rm -r "$MODEL_DATA_DIR"
fi
mkdir -p "$MODEL_DATA_DIR/original"
ln -sn "$DATASET_DIR" "$MODEL_DATA_DIR/original"
ln -sn "$TEST_DATASET_DIR/test/" "$MODEL_DATA_DIR/test"

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
