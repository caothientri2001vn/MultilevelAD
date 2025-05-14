from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import glob
import os

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
])
    # data_transforms = transforms.Compose([transforms.Normalize(),\
    #                 transforms.ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class ClassDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform, num_sample):
        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths = self.load_dataset(num_sample=num_sample)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, num_sample):
        img_paths = glob.glob(self.img_path + "/*.*")
        img_paths.sort()
        img_paths = img_paths[:num_sample]
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        ## simplex_noise
        return img,img_path.split('/')[-1]

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type

def load_data(dataset_name='mnist',normal_class=0,batch_size='16'):

    if dataset_name == 'cifar10':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'mnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'fashionmnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)


    elif dataset_name == 'retina':
        data_path = 'Dataset/OCT2017/train'

        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        # self.image_paths.sort()
    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type, img_path
    

    
class ClassDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, num_sample, class_name=None):
        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels, self.types = self.load_dataset(num_sample, class_name)  # self.labels => good : 0, anomaly : 1
        # self.image_paths.sort()
        
    def load_dataset(self, num_sample, class_name):
        img_tot_paths = []
        tot_labels = []
        tot_types = []

        # defect_types = os.listdir(self.img_path)
        if "example" in self.img_path:
            for level in range(5):
                if level == 0:
                    img_paths = glob.glob(self.img_path + '/level_' + str(level) + '_test/' + class_name + "/*.*")
                    img_paths.sort()
                    img_paths = img_paths[:num_sample]
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend([level] * len(img_paths))
                elif level == 4:
                    # print('ok')
                    img_paths = glob.glob(self.img_path + '/level_' + str(level) + "/flowers/*.*")
                    # print(len(img_paths))
                    img_paths.sort()
                    img_paths = img_paths[:num_sample]
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([level] * len(img_paths))
                else:
                    # print('ok')
                    img_paths = glob.glob(self.img_path + '/level_' + str(level) + "/*/*.*")
                    # print(len(img_paths))
                    img_paths.sort()
                    img_paths = img_paths[:num_sample]
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([level] * len(img_paths))

                    # img_tot_paths.extend(img_paths)
                    # tot_labels.extend([1] * len(img_paths))
                    # tot_types.extend([level] * len(img_paths))
        elif "diabetic-retinopathy" in self.img_path:
            for level in range(5):
                if level == 0:
                    img_paths = glob.glob(self.img_path + '/level_' + str(level) + '_test/*.*')
                    img_paths.sort()
                    img_paths = img_paths[:num_sample]
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend([level] * len(img_paths))
                else:
                    # print('ok')
                    img_paths = glob.glob(self.img_path + '/level_' + str(level) + "/*.*")
                    # print(len(img_paths))
                    img_paths.sort()
                    img_paths = img_paths[:num_sample]
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([level] * len(img_paths))
                


        return img_tot_paths, tot_labels, tot_types
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        

        return (img, label, img_type, img_path)
    
    

class GeneralDataset_test(torch.utils.data.Dataset):
    def __init__(self, transform, test_path_data, root):
        # self.img_path = root
        # self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        # load dataset
        self.img_paths, self.types = self.load_dataset(test_path_data, root)  # self.labels => good : 0, anomaly : 1
        # self.image_paths.sort()
        
    def load_dataset(self, test_path_data, root):
        df = pd.read_csv(test_path_data)
        columns_as_lists = {col: df[col].tolist() for col in df.columns}
        img_tot_paths = columns_as_lists['Path']
        img_tot_paths = [root + item for item in img_tot_paths]
        tot_types = columns_as_lists['Severity']


        return img_tot_paths, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, img_type = self.img_paths[idx], self.types[idx]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')

        ## Normal
        img = self.transform(img)
        ## simplex_noise
        label = int(img_type == 0)
        # print()
        return (img, label, img_type, img_path)