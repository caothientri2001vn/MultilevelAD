from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from dataset.noise import Simplex_CLASS
import cv2
import pandas as pd
class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image
    
    
class Normalize(object):
    """
    Only normalize images
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(),\
                    ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms



class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        # load dataset
        self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        # print(len(self.img_paths))

    def load_dataset(self):
        img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.*")
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img_normal = self.transform(img)
        ## simplex_noise
        size = 256
        h_noise = np.random.randint(10, int(size//8))
        w_noise = np.random.randint(10, int(size//8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256,256,3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)
        img_noise = img + init_zero
        img_noise = self.transform(img_noise)
        return img_normal,img_noise,img_path.split('/')[-1]


class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.simplexNoise = Simplex_CLASS()
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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img = self.transform(img)
        ## simplex_noise
        
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path)
    
    
    
    
    
class ClassDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform, num_sample):
        self.img_path = root
        self.simplexNoise = Simplex_CLASS()
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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img_normal = self.transform(img)
        ## simplex_noise
        size = 256
        h_noise = np.random.randint(10, int(size//8))
        w_noise = np.random.randint(10, int(size//8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256,256,3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)
        img_noise = img + init_zero
        img_noise = self.transform(img_noise)
        return img_normal,img_noise,img_path.split('/')[-1]
    
    
    
class ClassDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, num_sample, class_name=None):
        self.img_path = root
        self.simplexNoise = Simplex_CLASS()
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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img = self.transform(img)
        ## simplex_noise
        

        return (img, label, img_type, img_path)



class GeneralDataset_test(torch.utils.data.Dataset):
    def __init__(self, transform, test_path_data, root):
        # self.img_path = root
        self.simplexNoise = Simplex_CLASS()
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
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img = self.transform(img)
        ## simplex_noise
        label = int(img_type == 0)
        # print()
        return (img, label, img_type, img_path)