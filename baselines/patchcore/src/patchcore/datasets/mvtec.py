import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

import pandas as pd
import shutil


_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        dataset='mvtec',
        ood=False,
        ood_noise_type='',
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.dataset = dataset
        self.ood = ood
        self.ood_noise_type = ood_noise_type

        if self.ood:
            self.imgpaths_per_class, self.data_to_iterate = self.get_image_data_ood()
        else:
            self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)


    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:

            if 'mvtec' in self.source: # Hard code for now
                classpath = os.path.join(self.source, classname, self.split.value)
                maskpath = os.path.join(self.source, classname, "ground_truth")
                anomaly_types = os.listdir(classpath)

                imgpaths_per_class[classname] = {}
                maskpaths_per_class[classname] = {}

                for anomaly in anomaly_types:
                    anomaly_path = os.path.join(classpath, anomaly)
                    anomaly_files = sorted(os.listdir(anomaly_path))
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in anomaly_files
                    ]

                    if self.train_val_split < 1.0:
                        n_images = len(imgpaths_per_class[classname][anomaly])
                        train_val_split_idx = int(n_images * self.train_val_split)
                        if self.split == DatasetSplit.TRAIN:
                            imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                classname
                            ][anomaly][:train_val_split_idx]
                        elif self.split == DatasetSplit.VAL:
                            imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                classname
                            ][anomaly][train_val_split_idx:]

                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["good"] = None
                
            elif 'covid' in self.source or 'diabetic' in self.source or 'skin' in self.source or 'Novelty' in self.source: # Hard code for now

                classpath = self.source
                maskpath = ''

                imgpaths_per_class[classname] = {}
                maskpaths_per_class[classname] = {}

                if self.split.value == 'train':
                    anomaly_path = os.path.join(classpath, 'level_0_train')
                    anomaly = 'good'
                else:
                    anomaly_path = os.path.join(classpath, 'level_0_test')
                    anomaly = 'bad'
                    
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

            else:
                raise ValueError(f'Dataset {self.dataset} not supported')


        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


    def get_image_data_ood(self):
        imgpaths_per_class = {}

        for classname in self.classnames_to_use:
            if '_not' in classname: raise ValueError('Not implemented _not class')
            csv_template_folder = './data/template'

            if self.dataset.lower() == 'visa' or self.dataset.lower() == 'mvtec':
                df = pd.read_csv(os.path.join(csv_template_folder, f'{self.dataset.lower()}_{classname}_template.csv'))
                path_column = df['Path']
                path_list = path_column.tolist()
                path_list = [os.path.join(self.source, classname, 'test', x) for x in path_list]

            elif self.dataset.lower() == 'example':
                df = pd.read_csv(os.path.join(csv_template_folder, f'multidog_{classname}_template.csv'))
                path_column = df['Path']
                path_list = path_column.tolist()
                path_list = [os.path.join(self.source, x) for x in path_list]

            elif self.dataset.lower() == 'diabetic_retinopathy'.replace('-', '_'):
                assert len(self.classnames_to_use) == 1, "Only one class is allowed for diabetic_retinopathy"
                df = pd.read_csv(os.path.join(csv_template_folder, 'diabetic-retinopathy_template.csv'))
                path_column = df['Path']
                path_list = path_column.tolist()
                path_list = [os.path.join(self.source, x) for x in path_list]

            elif self.dataset.lower() == 'covid19':
                assert len(self.classnames_to_use) == 1, "Only one class is allowed for covid19"
                df = pd.read_csv(os.path.join(csv_template_folder, 'covid19_template.csv'))
                path_column = df['Path']
                path_list = path_column.tolist()
                path_list = [os.path.join(self.source, x) for x in path_list]
            
            elif self.dataset.lower() == 'skin_lesion':
                assert len(self.classnames_to_use) == 1, "Only one class is allowed for skin_lesion"
                df = pd.read_csv(os.path.join(csv_template_folder, 'skin-lesion_template.csv'))
                path_column = df['Path']
                path_list = path_column.tolist()
                path_list = [os.path.join(self.source, x) for x in path_list]
            
            else:
                raise ValueError(f'Dataset {self.dataset} not supported')

            if classname not in imgpaths_per_class:
                imgpaths_per_class[classname] = {'bad': path_list}
            else:
                imgpaths_per_class[classname]['bad'].extend(path_list)


        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for i, image_path in enumerate(imgpaths_per_class[classname]['bad']):
                data_tuple = [classname, 'bad', image_path]
                data_tuple.append(None)  # mask_path
                data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
