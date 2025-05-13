#!/usr/bin/env bash
set -e
exec > output_skin.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

DATASETS=("skin")
# DATASETS=("carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "toothbrush" "transistor" "zipper")
# DATASETS=("bottle")

MODEL_DATA_DIR="./dataset_skin"
MODEL_OUTPUT_DIR="./result_skin"

for DATASET_NAME in "${DATASETS[@]}"
do
DATASET_DIR="$(pwd)/../../data/severity-based/skin-lesion"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test/good"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test/bad"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/train"

ln -sn "$DATASET_DIR/level_0_train" "$MODEL_DATA_DIR/$DATASET_NAME/train/good"

find "$DATASET_DIR/level_0_test" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/$DATASET_NAME/test/good/$(basename "$file")"
done

find "$DATASET_DIR/NV_level_1" -maxdepth 1 -type f | sort | head -n 50 | while IFS= read -r file; do
  ln -sn "$file" "$MODEL_DATA_DIR/$DATASET_NAME/test/bad/$(basename "$file")"
done

echo "Training on dataset: $DATASET_NAME"
CUDA_VISIBLE_DEVICES=3 python -u train_coreset_distribution.py \
    --category $DATASET_NAME \
    --seed 23 \
    --train_coreset \
    --train_nb_dist \
    --train_coor_dist \
    --dataset_path "$MODEL_DATA_DIR" \
    --project_root_path "$MODEL_OUTPUT_DIR"

# echo "Calculating scores on dataset: $DATASET_NAME"
# CUDA_VISIBLE_DEVICES=6 python -u analysis_code/calc_ensemble_score.py \
#     --category $DATASET_NAME \
#     --backbone_list WR101 \
#     --project_root_path "$MODEL_OUTPUT_DIR" \
#     --ensemble_root_path "$MODEL_OUTPUT_DIR/WR101_result"

done

# # convert result format and save it into "./result/ensemble_ravel" repository.
# # Add argument "--is_BTAD" if dataset is BTAD, and "--is_MVtec_small" if dataset is small version of MVTec which we provided.
# # Default dataste is MVTec AD benchmark.
# python analysis_code/convert_result_format.py --before_result_root_dir ./result/WR101_result --after_result_root_dir ./result/WR101_ravel

# # analysis anomaly map from "./result/ensemble_ravel" repository.
# # Add argument "--visualize" to visualize anomaly map on "./result/ensemble_ravel/viz" repository.
# # If you want to find misclassified images with trained model, add argument "--calc_misclassified_sample" and indices of false positive samples and false negative samples will be presented on "./result/ensemble_ravel/misclassified_sample_list.csv"
# # In addition, add "--calc_pro" argument to additionally calculate AUPRO score. The result will presented on "./result/ensemble_ravel/score_result.csv".
# python analysis_code/analysis_amap.py --project_root_path ./result/WR101_ravel --visualize