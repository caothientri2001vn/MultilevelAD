# Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection

## 0. Dataset
### To download the dataset, just simply run:
```
import os
from datasets import load_dataset

ds = load_dataset("tricao1105/MultilevelADSet")  

output_root = "./MultilevelAD/data"  

for sample in ds:
    image = sample["image"]
    rel_path = sample["relative_path"]
    
    save_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
```
### Data structure:
After download, the dataset will be stored in the following structure:
```
MAD-Bench/
├── Medical/
│   ├── covid19/
│   ├── diabetic-retinopathy/
│   └── skin-lesion/
│
├── Industry/
│   ├── mvtec/
│   └── visa/
│
└── NoveltyClass/
    └── multidog/
```
