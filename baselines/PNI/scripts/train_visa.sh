#!/usr/bin/env bash
set -e
exec > output_visa.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PNI

DATASETS=("capsules" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pipe_fryum")
# DATASETS=("carpet" "grid" "leather" "tile" "wood" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "toothbrush" "transistor" "zipper")
# DATASETS=("bottle")

MODEL_DATA_DIR="./dataset_visa"
MODEL_OUTPUT_DIR="./result_visa"

for DATASET_NAME in "${DATASETS[@]}"
do
DATASET_DIR="$(pwd)/../../data/area-based/VisA_reorganized/$DATASET_NAME"

rm -rf "$MODEL_DATA_DIR"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/ground_truth"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/test"
mkdir -p "$MODEL_DATA_DIR/$DATASET_NAME/train"
ln -sn "$DATASET_DIR/ground_truth/bad" "$MODEL_DATA_DIR/$DATASET_NAME/ground_truth/bad"
ln -sn "$DATASET_DIR/test/bad" "$MODEL_DATA_DIR/$DATASET_NAME/test/bad"
ln -sn "$DATASET_DIR/test/good_level_0" "$MODEL_DATA_DIR/$DATASET_NAME/test/good"
ln -sn "$DATASET_DIR/train/good" "$MODEL_DATA_DIR/$DATASET_NAME/train/good"

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