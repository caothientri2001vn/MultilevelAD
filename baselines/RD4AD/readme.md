## Environment
```bash
pip install -r requirements.txt
```
## Train 

### MVTec
Run:
```
python main_industry.py --dataset_name mvtec
```
### VisA
Run:
```
python main_industry.py --dataset_name visa
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
After running evalution, the results will be stored in `./MultilevelAD/results/RD4AD/`
    
## Acknowledgement
This implementation is adapted and modified based on the original [RD4AD](https://github.com/hq-deng/RD4AD/) code. We are thankful to their brilliant works!
