import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
import os
from PIL import Image

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=True, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

class ADDataset:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = []
        self.labels = []
        for subdir in os.listdir(root):
            for image in os.listdir(os.path.join(root, subdir)):
                self.paths.append(os.path.join(root, subdir, image))
                self.labels.append(int(subdir not in ['good', 'good_level_0']))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class ADTestDataset:
    def __init__(self, root, in_file, transform=None):
        self.root = root
        self.transform = transform
        with open(in_file, 'r') as f:
            self.paths = f.readlines()
        self.paths = [path.strip() for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.paths[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]

def get_loaders(dataset_dir, batch_size):
    transform = transform_color
    trainset = ADDataset(os.path.join(dataset_dir, 'train'), transform)
    testset = ADDataset(os.path.join(dataset_dir, 'test'), transform)

    print(len(trainset))
    print(len(testset))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, test_loader

def get_test_loaders(dataset_dir, batch_size, dataset, in_file):
    transform = transform_color
    trainset = ADDataset(os.path.join(dataset_dir, 'original', dataset, 'train'), transform)
    testset = ADTestDataset(os.path.join(dataset_dir, 'test'), in_file, transform)

    print(len(trainset))
    print(len(testset))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, test_loader

def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)


