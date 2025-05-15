#!/usr/bin/env bash
set -e
exec > logs/infer_mvtec_${2}_$1.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

if [ "$1" == "a" ]; then
    DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable")
else
    DATASETS=("capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")
fi

# DATASETS=("bottle" "carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "transistor" "zipper")

INPUT_FILE_NAME="input_list/mvtec_${2}_${1}.txt"
MODEL_DATA_DIR="./dataset_mvtec_${2}_${1}"
MODEL_OUTPUT_DIR="./result_mvtec"

for DATASET_NAME in "${DATASETS[@]}"
do
OUTPUT_FILE_NAME="csv_corrupted/pni_mvtec_${DATASET_NAME}_corrupted_$2.csv"
DATASET_DIR="$(pwd)/../../data/area-based/mvtec/$DATASET_NAME"
TEST_DATASET_DIR="$(pwd)/../../data/corrupted/$2/mvtec_order/$DATASET_NAME"

if [[ -d "$MODEL_DATA_DIR" ]]; then
  rm -r "$MODEL_DATA_DIR"
fi
mkdir -p "$MODEL_DATA_DIR/original"
ln -sn "$DATASET_DIR" "$MODEL_DATA_DIR/original/$DATASET_NAME"
ln -sn "$TEST_DATASET_DIR/test/" "$MODEL_DATA_DIR/test"

python csv_to_txt.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$INPUT_FILE_NAME"

echo "Infering on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=$3 python -u infer.py \
    --category $DATASET_NAME \
    --seed 23 \
    --dataset_path "$MODEL_DATA_DIR/original" \
    --project_root_path "$MODEL_OUTPUT_DIR" \
    --in_file "$INPUT_FILE_NAME" \
    --out_file "$OUTPUT_FILE_NAME" \
    --test_dir "$MODEL_DATA_DIR/test"

python merge_csv.py "$TEST_DATASET_DIR/${DATASET_NAME}_template.csv" "$OUTPUT_FILE_NAME" "merged_$OUTPUT_FILE_NAME"
done
