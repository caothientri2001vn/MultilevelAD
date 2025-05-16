## Environments 
- Python : 3.8
- CUDA   : 11.3
## Packages
```bash
cd Score-based
pip install -r requirements.txt
```
## Pretrained weights
We use the available pretrained weights from the author, which can be found in. Download pretrained weights from [[Google Drive]](https://drive.google.com/drive/folders/1fvF1RFeOCWIraWhTUu71ZX1TX5Za8_kb?usp=drive_link). To download the weights, run:
```bash
python checkpoints_download.py
```
## Train
### Covid19
```bash
python train_severity.py \
    --dataset_path ../data/severity-based/covid19 \
    --save_path ./save/ \
    --class_name all
```
### Diabetic Retinopathy 
``` bash
python train_dia.py \
    --dataset_path ../data/severity-based/diabetic-retinopathy \
    --save_path ./save/ \
    --class_name all
```
### Skin Lesion
``` bash
python train_skin.py \
    --dataset_path ../data/severity-based/skin-lesion \
    --save_path ./save/ \
    --class_name all
```
### VisA (Area-based)
``` bash
python train_visa.py \
    --dataset_path ../data/area-based/VisA_reorganized \
    --save_path ./save/ \
    --class_name all
```
### Class-based Example (Level 0)
``` bash
python train.py \
    --dataset_path ../data/class-based/example/level_0_train \
    --save_path ./save/ \
    --class_name all
```
## Evaluate
### Covid19
```bash
python inference_covid19.py
```
### Diabetic Retinopathy 
``` bash
python inference_dia.py
```
### VisA (Area-based)
``` bash
python inference_visa.py
```
### Class-based Example (Level 0)
``` bash
python inference_example.py
```
After running evalution, the results will be stored in `../results/`
## Acknowledgement
This implementation is adapted and modified based on the original [Anomaly Detection using Score-based Perturbation Resilience](https://github.com/Lee-JongHyeon/Anomaly-Detection-using-Score-based-Perturbation-Resilience) code. We are thankful to their brilliant works!




