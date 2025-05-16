import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
import os

def get_score(model, device, train_loader, test_loader, out_file):
    model.eval()
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        test_paths = []
        for (imgs, paths) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            if features.ndim == 1:
                features = features.unsqueeze(0)
            test_feature_space.append(features)
            test_paths.extend(paths)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    with open(out_file, 'w') as f:
        f.write('Path,Anomaly Score\n')
        for path, distance in zip(test_paths, distances):
            f.write(f'"{path}",{distance}\n')

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.dataset, 'latest.pth'), weights_only=False))

    train_loader, test_loader = utils.get_test_loaders(dataset_dir=args.dataset_dir, batch_size=args.batch_size, dataset=args.dataset, in_file=args.in_file)
    get_score(model, device, train_loader, test_loader, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--in_file')
    parser.add_argument('--out_file')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    main(args)
