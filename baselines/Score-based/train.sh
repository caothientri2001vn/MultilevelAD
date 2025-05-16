# python train.py --dataset_path /home/tri/multi-level-anomaly/data/class-based/example/level_0_train    \
#                --save_path ./save/        \
#                --class_name all

# python train_severity.py --dataset_path /home/tri/multi-level-anomaly/data/severity-based/covid19    \
#                --save_path ./save/        \
#                --class_name all

# python train_severity.py --dataset_path /home/tri/multi-level-anomaly/data/severity-based/diabetic-retinopathy    \
#                --save_path ./save/        \
#                --class_name all

# python train_visa.py --dataset_path /home/tri/multi-level-anomaly/data/area-based/VisA_reorganized    \
#                --save_path ./save/        \
#                --class_name all

# python train_dia.py --dataset_path /home/tri/multi-level-anomaly/data/severity-based/diabetic-retinopathy    \
#                --save_path ./save/        \
#                --class_name all

python train_skin.py --dataset_path /home/tri/multi-level-anomaly/data/severity-based/skin-lesion    \
               --save_path ./save/        \
               --class_name all