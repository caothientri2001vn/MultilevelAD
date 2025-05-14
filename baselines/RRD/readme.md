## Environment
```bash
pip install -r requirements.txt
```
## Train
### MVTec
We use the available pretrained weights from the author, which can be found in [[Google Drive]](https://drive.google.com/drive/folders/1ifrkexB0N1O87CpYPS-Wg2vgAiwXFf2Z). The weights should be placed in `./checkpoint/mvtec/`
### VisA
Run:
```
python main_visa.py
```
### Multidog
Run:
```
python main_multidog.py
```
### Skin-lesion
Run:
```
python main_medical.py --dataset_name skin-lesion
```
### Diabetic-Retinopathy
Run:
```
python main_medical.py --dataset_name diabetic-retinopathy
```
### Covid19
Run:
```
python main_medical.py --dataset_name covid19
```
All the checkpoint will be in `./checkpoint`
## Evaluate
Run:
```
python inference_general.py
```
After running evalution, the results will be stored in `./MultilevelAD/results/RRD/`
## Acknowledgement
This implementation is adapted and modified based on the original [RRD](hhttps://github.com/tientrandinh/Revisiting-Reverse-Distillation/) code. We are thankful to their brilliant works!
