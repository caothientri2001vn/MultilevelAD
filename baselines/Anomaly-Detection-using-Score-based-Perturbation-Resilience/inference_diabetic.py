import mvtec
from mvtec import MVTecDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve

import pandas as pd
import numpy as np
import functools
import os
import random
import argparse
import warnings
import gc
from tqdm import tqdm
from unet import UNet
from torch_ema import ExponentialMovingAverage
from test import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inference(class_name, path_test_data, root):
    batch_size = 1
    num_workers = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_dataset   = MVTecDataset(dataset_path      = path_test_data, 
                                      class_name    =  class_name, 
                                      is_train      =  False,
                                      root          = root)

    test_loader    = DataLoader(dataset         =   test_dataset, 
                                batch_size      =   batch_size, 
                                pin_memory      =   True,
                                shuffle         =   False,
                                drop_last       =   False,
                                num_workers     =   num_workers)
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma = 25, device = device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma = 25, device = device)
    
    score_model = UNet(marginal_prob_std = marginal_prob_std_fn,
                        n_channels        = 3,
                        n_classes         = 3,
                        embed_dim         = 256)
    load_pth = '/home/tri/llms-anomaly-detection/score-based/save/models/level_0_train_dia.pth'
    ckpt = torch.load(load_pth, map_location=device)
    score_model = score_model.to(device)
    score_model.load_state_dict(ckpt)
    score_model.eval()
    
    all_scores = None
    all_mask = None
    all_x = None
    all_y = None
    paths = []
    levels = []

    for x, y, mask, path, level in tqdm(test_loader):
        path_ = path[0]
        real_path = "/".join(path_.split('/')[-2:])
        paths.append(real_path)
        levels.append(level.to('cpu').detach().numpy()[0])

        x = x.to(device)
        sample_batch_size = x.shape[0]
        perturbed_t = 1e-3
        t = torch.ones(sample_batch_size, device=device) * perturbed_t

        scores = 0.
        num_iter = 3
        with torch.no_grad():
            for i in range(num_iter):
                ms = marginal_prob_std_fn(t)[:, None, None, None]
                g = diffusion_coeff_fn(t)[:, None, None, None]
                n = torch.randn_like(x)*ms
                z = x + n
                score = score_model(z, t)
                score = score*ms**2 + n
                scores += (score**2).mean(1, keepdim = True)
        scores /= num_iter

        all_scores = torch.cat((all_scores, scores), dim = 0) if all_scores != None else scores
        all_mask = torch.cat((all_mask,mask), dim = 0) if all_mask != None else mask
        all_x = torch.cat((all_x,x), dim = 0) if all_x != None else x
        all_y = torch.cat((all_y,y), dim = 0) if all_y != None else y

    heatmaps = all_scores.cpu().detach().sum(1, keepdim = True)
    heatmaps = F.interpolate(torch.Tensor(heatmaps), (256, 256), mode = "bilinear", align_corners=False)
    heatmaps = F.avg_pool2d(heatmaps, 31,1, padding = 15).numpy()
    # import pdb; pdb.set_trace()
    sums = []
    maxs = []

    sums = heatmaps.sum(axis=(1,2,3))
    maxs = heatmaps.max(axis=(1,2,3))
    img_scores = maxs.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    # for i in range(heatmaps.shape[0]):
    #     sample = heatmaps[i]
    #     # sample_sum = np.sum(sample)
    #     # sample_max = np.max(sample)
    #     # sample_sum = heatmap.su
    #     # sample_max = heatmaps.max(axis=(1,2,3))

    #     sums.append(sample_sum)
    #     maxs.append(sample_max)

    # df = pd.DataFrame({
    #     'sum': sums,
    #     'max': maxs
    # })
    
    return sums, maxs, paths, levels

if __name__ == '__main__':
    root_general = '/home/tri/multi-level-anomaly/data/severity-based/diabetic-retinopathy/'
    path_test_data = '/home/tri/multi-level-anomaly/data/template/' + 'diabetic' + '_template.csv' 
    root = root_general
    sums, maxs, paths, levels = inference(class_name='diabetic', path_test_data=path_test_data, root=root)
    df = pd.DataFrame({
        'Path': paths,
        'Area': levels,
        'Anomaly Score': maxs,
        'Anomaly Score Sum': sums
    })
    df.to_csv('/home/tri/llms-anomaly-detection/new/Anomaly-Detection-using-Score-based-Perturbation-Resilience/results/score-based_' + 'diabetic' + '.csv', index=False)