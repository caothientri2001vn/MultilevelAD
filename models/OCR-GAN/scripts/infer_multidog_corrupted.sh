#!/usr/bin/env bash
set -e
exec > logs/infer_multidog_${2}.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

# if [ "$1" == "a" ]; then
#     DATASETS=("bichon_frise" "chinese_rural_dog" "golden_retriever")
# else
#     DATASETS=("labrador_retriever" "teddy")
# fi

DATASETS=("bichon_frise" "chinese_rural_dog" "golden_retriever" "labrador_retriever" "teddy")

INPUT_FILE_NAME="input_list/multidog_${2}.txt"
MODEL_DATA_DIR="./dataset_multidog_${2}"
MODEL_OUTPUT_DIR="./output_32px_multidog_$2"

for DATASET_NAME in "${DATASETS[@]}"
do
OUTPUT_FILE_NAME="csv_corrupted/ocr-gan-32px_example_${DATASET_NAME}_corrupted_$2.csv"
DATASET_DIR="$(pwd)/../../data/class-based/example"
TEST_DATASET_DIR="$(pwd)/../../data/corrupted/$2/example"

if [[ -d $MODEL_DATA_DIR ]]; then
  rm -r "$MODEL_DATA_DIR"
fi
mkdir -p "$MODEL_DATA_DIR/train/good"
mkdir -p "$MODEL_DATA_DIR/test/good"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sn "$TEST_DATASET_DIR" "$MODEL_DATA_DIR/real_test"

# Process each file in the subdirectory
find "$DATASET_DIR/level_0_train/$DATASET_NAME" -maxdepth 1 -type f | sort | head -n 500 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/train/good/$(basename "$file")"
done

find "$DATASET_DIR/level_0_test/$DATASET_NAME" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/good/$(basename "$file")"
done

find "$DATASET_DIR/level_1/other_dogs" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/test/bad/$(basename "$file")"
done

python csv_to_txt.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

CUDA_VISIBLE_DEVICES=$3 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --niter 50 \
    --outf "$MODEL_OUTPUT_DIR" \
    --testroot "$MODEL_DATA_DIR/real_test" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME"

python merge_csv.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
