#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate skipganomaly

DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")

INPUT_FILE_NAME="test_input_files_visa.txt"
MODEL_DATA_DIR="./data_visa"

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Infering on dataset: $DATASET_NAME"

OUTPUT_FILE_NAME="result_csvs/skip-gan_visa_${DATASET_NAME}.csv"
DATASET_DIR="$(pwd)/../../data/Industry/visa/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
ln -sn "$DATASET_DIR/test/" "$MODEL_DATA_DIR"

python csv_to_txt.py "$(pwd)/../../data/template/visa_${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=7 python infer.py \
    --model skipganomaly \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --manualseed 1 \
    --load_weights \
    --in_file "$INPUT_FILE_NAME" \
    --outf "$(pwd)/output_visa" \
    --out_file "$OUTPUT_FILE_NAME"
    # --save_image_freq 5
    # --save_test_images \
    # --workers 8 \
    # --nc 3 \
    # --device gpu \
    # --beta1 0.5 \
    # --w_adv 1 \
    # --w_con 50 \
    # --w_lat 1

python merge_csv.py "$(pwd)/../../data/template/visa_${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
