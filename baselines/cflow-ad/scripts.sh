############################################################ Train ############################################################
### Medical datasets
# Dataset covid19
python3 main.py --gpu 0 --pro -inp 256 --dataset covid19 --class-name covid

# Dataset diabetic_retinopathy
python3 main.py --gpu 0 --pro -inp 256 --dataset diabetic_retinopathy --class-name diabetic_retinopathy

# Dataset skin_lesion
python3 main.py --gpu 0 --pro -inp 256 --dataset skin_lesion --class-name skin_lesion


### Industry datasets
# Dataset mvtec
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name bottle
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name cable
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name capsule
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name carpet
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name grid
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name hazelnut
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name leather
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name metal_nut
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name pill
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name screw
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name tile
python3 main.py --gpu 0 --pro -inp 128 --dataset mvtec --class-name transistor
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name wood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name zipper

# Dataset VisA
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name capsules
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name chewinggum
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name fryum
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name macaroni1
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name macaroni2
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pcb1
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pcb2
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pcb3
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pipe_fryum


### NoveltyClass dataset
# NoveltyClass dataset
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name bichon_frise
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name chinese_rural_dog
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name golden_retriever
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name labrador_retriever
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name teddy




############################################################ Eval ############################################################
### Medical datasets
# Dataset covid19
python3 main.py --gpu 0 --pro -inp 256 --dataset covid19 --class-name covid --action-type norm-test --checkpoint PATH/FILE.PT --ood

# Dataset diabetic_retinopathy
python3 main.py --gpu 0 --pro -inp 256 --dataset diabetic_retinopathy --class-name diabetic_retinopathy --action-type norm-test --checkpoint PATH/FILE.PT --ood

# Dataset skin_lesion
python3 main.py --gpu 0 --pro -inp 256 --dataset skin_lesion --class-name skin_lesion --action-type norm-test --checkpoint PATH/FILE.PT --ood


### Industry datasets
# Dataset mvtec
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name bottle --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name cable --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name capsule --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name carpet --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name grid --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name hazelnut --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name leather --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name metal_nut --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name pill --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name screw --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name tile --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 128 --dataset mvtec --class-name transistor --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name wood --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name zipper --action-type norm-test --checkpoint PATH/FILE.PT --ood

# Dataset VisA
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name capsules --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name chewinggum --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name fryum --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name macaroni1 --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name macaroni2 --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pcb1 --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pcb2 --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pcb3 --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset VisA --class-name pipe_fryum --action-type norm-test --checkpoint PATH/FILE.PT --ood


### NoveltyClass dataset
# NoveltyClass dataset
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name bichon_frise --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name chinese_rural_dog --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name golden_retriever --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name labrador_retriever --action-type norm-test --checkpoint PATH/FILE.PT --ood
python3 main.py --gpu 0 --pro -inp 256 --dataset example --class-name teddy --action-type norm-test --checkpoint PATH/FILE.PT --ood
