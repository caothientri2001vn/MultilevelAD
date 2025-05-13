#!/usr/bin/env bash
set -e
exec > logs/infer_mvtec_${2}.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate OCR-GAN

# if [ "$1" == "a" ]; then
#     DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable")
# else
#     DATASETS=("capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")
# fi

DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")

INPUT_FILE_NAME="input_list/mvtec_${2}.txt"
MODEL_DATA_DIR="./dataset_mvtec_${2}"
MODEL_OUTPUT_DIR="./output_32px_$2"

for DATASET_NAME in "${DATASETS[@]}"
do
OUTPUT_FILE_NAME="csv_corrupted/ocr-gan-32px_mvtec_${DATASET_NAME}_corrupted_$2.csv"
DATASET_DIR="$(pwd)/../../data/area-based/mvtec_order/$DATASET_NAME"
TEST_DATASET_DIR="$(pwd)/../../data/corrupted/$2/mvtec_order/$DATASET_NAME"

rm -r "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/train/"
mkdir -p "$MODEL_DATA_DIR/test/bad"

ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/train/good"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/test/good"
ln -sn "$TEST_DATASET_DIR/test" "$MODEL_DATA_DIR/real_test"

# Process all other subdirectories of test
for subdir in "$DATASET_DIR/test"/*; do
  if [ -d "$subdir" ] && [ "$(basename "$subdir")" != "good_level_0" ]; then
    subdir_name=$(basename "$subdir")

    # Process each file in the subdirectory
    for file in "$subdir"/*; do
      if [ -f "$file" ]; then
        ln -sn "$file" "$MODEL_DATA_DIR/test/bad/${subdir_name}_$(basename "$file")"
      fi
    done
  fi
done

python csv_to_txt.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=$3 python train_all.py \
    --model ocr_gan_aug \
    --dataset $DATASET_NAME \
    --dataroot "$MODEL_DATA_DIR" \
    --isize 32 \
    --outf "$MODEL_OUTPUT_DIR" \
    --niter 50 \
    --testroot "$MODEL_DATA_DIR/real_test" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME"

python merge_csv.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
