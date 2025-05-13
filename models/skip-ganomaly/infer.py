"""
Infer SKIP/GANOMALY
"""
from options import Options

## dataloader.py
"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

class FileListDataset(Dataset):
    def __init__(self, opt, transform):
        super().__init__()
        self.opt = opt
        with open(self.opt.in_file, 'r') as f:
            self.files = f.readlines()
        self.files = [line.strip() for line in self.files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.opt.dataroot, self.files[idx])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.files[idx]

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    # test_ds = ImageFolder(opt.dataroot, transform)

    test_ds = FileListDataset(opt, transform)

    ## DATALOADER
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(None, None, test_dl)


from lib.models import load_model

##
def main():
    """ Testing
    """
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    model.test_to_file()

if __name__ == '__main__':
    main()
