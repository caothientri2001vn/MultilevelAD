#!/usr/bin/env bash
set -e
exec > output_infer.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate skipganomaly

DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")
# DATASETS=("bottle")

INPUT_FILE_NAME="test_input_files.txt"
MODEL_DATA_DIR="./data"

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Infering on dataset: $DATASET_NAME"

OUTPUT_FILE_NAME="an_scores_${DATASET_NAME}.csv"
DATASET_DIR="$(pwd)/../../data/area-based/mvtec_order/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
ln -sfn "$DATASET_DIR/test/" "$MODEL_DATA_DIR"

cut -d',' -f1 "$DATASET_DIR/${DATASET_NAME}_template.csv" | tail -n +2 > "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=7 python infer.py \
    --model skipganomaly \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 256 \
    --manualseed 1 \
    --load_weights \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME"
    # --save_image_freq 5
    # --save_test_images \
    # --workers 8 \
    # --nc 3 \
    # --device gpu \
    # --outf "$(pwd)/output" \
    # --beta1 0.5 \
    # --w_adv 1 \
    # --w_con 50 \
    # --w_lat 1

python merge_csv.py "$DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
