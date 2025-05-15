#!/usr/bin/env bash
set -e
exec > output_visa_32px.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

INPUT_FILE_NAME="test_input_files_visa_32px.txt"
MODEL_DATA_DIR="./data_visa_32px"
MODEL_OUTPUT_DIR="./output_visa_32px/"
DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")
# DATASETS=("hazelnut")

for DATASET_NAME in "${DATASETS[@]}"
do
echo "Training on dataset: $DATASET_NAME"

DATASET_DIR="$(pwd)/../../data/area-based/VisA_reorganized/$DATASET_NAME"
OUTPUT_FILE_NAME="csv_rerun/ocr-gan-32px_visa_${DATASET_NAME}.csv"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train/"
mkdir -p "$MODEL_DATA_DIR/test/"

ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/train/good"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/test/good"
ln -sn "$DATASET_DIR/test/bad" "$MODEL_DATA_DIR/test/bad"
ln -sn "$DATASET_DIR/test" "$MODEL_DATA_DIR/real_test"

python csv_to_txt.py "$DATASET_DIR/${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=2 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --niter 50 \
    --outf "$MODEL_OUTPUT_DIR" \
    --testroot "$MODEL_DATA_DIR/real_test" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME"

python merge_csv.py "$DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
