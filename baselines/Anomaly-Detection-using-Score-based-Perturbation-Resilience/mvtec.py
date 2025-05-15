import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# CLASS_NAMES  = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#                 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#                 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# /home/tri/multi-level-anomaly/data/class-based/example/level_0_test/golden_retriever
# CLASS_NAMES = ['bichon_frise', 'chinese_rural_dog', 'golden_retriever', 'labrador_retriever', 'teddy']
# CLASS_NAMES = ['teddy']
# CLASS_NAMES = ['chinese_rural_dog', 'golden_retriever', 'labrador_retriever']
class MVTecDataset(Dataset):
    def __init__(self, dataset_path, class_name='bottle', is_train=True, root='a'):
        # assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.root = root
        # self.is_train = False

        self.x, self.y, self.levels = self.load_dataset_folder()
        
        self.transform_x    =   T.Compose([T.Resize(192, Image.LANCZOS),
                                           T.ToTensor()])
        self.transform_x = T.Compose([
            T.Resize(256, interpolation=InterpolationMode.LANCZOS),
            # T.CenterCrop(size=(192, 192)),
            T.ToTensor()
        ])

        self.transform_mask = T.Compose([
            T.Resize(size=256, interpolation=InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        img_path = self.x[idx]
        level = self.levels[idx]
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x).convert('RGB')
            
        x = self.transform_x(x)
        
        x = 2 * x - 1.
        
        return x, y, img_path, level

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        # import pdb; pdb.set_trace()
        # phase = 'train' if self.is_train else 'test'
        # import pdb; pdb.set_trace()
        x, y, level = [], [], []

        df = pd.read_csv(self.dataset_path)
        columns_as_lists = {col: df[col].tolist() for col in df.columns}
        img_tot_paths = columns_as_lists['Path']
        img_tot_paths = [self.root + item for item in img_tot_paths]
        x = img_tot_paths
        levels = df['Severity'].tolist()
        # good 
        y.extend([0] * len(x))
        # mask.extend([None] * len(x))

        assert len(x) == len(y), 'Number of x and y should be the same.'

        return list(x), list(y), list(levels)
