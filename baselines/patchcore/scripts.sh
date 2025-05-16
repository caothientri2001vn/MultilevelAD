############################################################ Train ############################################################
### Medical datasets
# Dataset covid19
datapath=data/Medical/covid19
datasets=('covid')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python3 bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group covid19_B32_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 --batch_size 32 "${dataset_flags[@]}" mvtec $datapath

# Dataset diabetic_retinopathy
datapath=data/Medical/diabetic-retinopathy
datasets=('diabetic_retinopathy')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python3 bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group diabetic_retinopathy_B32_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 --batch_size 32 "${dataset_flags[@]}" mvtec $datapath

# Dataset skin_lesion
datapath=data/Medical/skin-lesion
datasets=('skin_lesion')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python3 bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group skin_lesion_B32_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 --batch_size 32 "${dataset_flags[@]}" mvtec $datapath


### Industry datasets
# Dataset mvtec
datapath=data/Industry/mvtec
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python3 bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath

# Dataset VisA
datapath=data/Industry/visa
datasets=('capsules' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pipe_fryum')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python3 bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group VisA_B8_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 --batch_size 32 "${dataset_flags[@]}" mvtec $datapath


### NoveltyClass dataset
# NoveltyClass dataset
datapath=data/NoveltyClass/multidog
datasets=('bichon_frise' 'chinese_rural_dog' 'golden_retriever' 'labrador_retriever' 'teddy')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python3 bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group example_B32_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 --batch_size 32 "${dataset_flags[@]}" mvtec $datapath



############################################################ Eval ############################################################
### Medical datasets
# Dataset covid19
datapath=data/Medical/covid19
loadpath=./results/MVTecAD_Results
modelfolder=<model_folder_inside_results_folder>
savefolder=evaluated_results'/'$modelfolder
datasets=('covid')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
python bin/load_and_evaluate_patchcore.py --dataset 'covid19' --gpu 0 --seed 0 $savefolder patch_core_loader "${model_flags[@]}" --faiss_on_gpu dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --ood covid19 $datapath

# Dataset diabetic_retinopathy
datapath=data/Medical/diabetic-retinopathy
loadpath=./results/MVTecAD_Results
modelfolder=<model_folder_inside_results_folder>
savefolder=evaluated_results'/'$modelfolder
datasets=('diabetic_retinopathy')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
python bin/load_and_evaluate_patchcore.py --dataset 'diabetic_retinopathy' --gpu 0 --seed 0 $savefolder patch_core_loader "${model_flags[@]}" --faiss_on_gpu dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --ood diabetic_retinopathy $datapath

# Dataset skin_lesion
datapath=data/Medical/skin-lesion
loadpath=./results/MVTecAD_Results
modelfolder=<model_folder_inside_results_folder>
savefolder=evaluated_results'/'$modelfolder
datasets=('skin_lesion')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
python bin/load_and_evaluate_patchcore.py --dataset 'skin_lesion' --gpu 0 --seed 0 $savefolder patch_core_loader "${model_flags[@]}" --faiss_on_gpu dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --ood skin_lesion $datapath


### Industry datasets
# Dataset mvtec
datapath=data/Industry/mvtec
loadpath=./results/MVTecAD_Results
modelfolder=<model_folder_inside_results_folder>
savefolder=evaluated_results'/'$modelfolder
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'transistor' 'wood' 'zipper')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
python bin/load_and_evaluate_patchcore.py --dataset 'mvtec' --gpu 0 --seed 0 $savefolder patch_core_loader "${model_flags[@]}" --faiss_on_gpu dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --ood mvtec $datapath

# Dataset VisA
datapath=data/Industry/visa
loadpath=./results/MVTecAD_Results
modelfolder=<model_folder_inside_results_folder>
savefolder=evaluated_results'/'$modelfolder
datasets=('capsules' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pipe_fryum')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
python bin/load_and_evaluate_patchcore.py --dataset 'vias' --gpu 0 --seed 0 $savefolder patch_core_loader "${model_flags[@]}" --faiss_on_gpu dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --ood visa $datapath


### NoveltyClass dataset
# NoveltyClass dataset
datapath=data/NoveltyClass/multidog
loadpath=./results/MVTecAD_Results
modelfolder=<model_folder_inside_results_folder>
savefolder=evaluated_results'/'$modelfolder
datasets=('bichon_frise' 'chinese_rural' 'golden_retriever' 'labrador_retriever' 'teddy')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
python bin/load_and_evaluate_patchcore.py --dataset 'example' --gpu 0 --seed 0 $savefolder patch_core_loader "${model_flags[@]}" --faiss_on_gpu dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --ood example $datapath

