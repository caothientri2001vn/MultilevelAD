from __future__ import print_function
import os, random, time, math
import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from config import get_args

from custom_datasets import *


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(c):
    
    # Model
    if c.action_type in ['norm-train', 'norm-test']:
        c.model = "{}_{}_{}_pl{}_cb{}_inp{}_run{}_{}".format(
            c.dataset, c.enc_arch, c.dec_arch, c.pool_layers, c.coupling_blocks, c.input_size, c.run_name, c.class_name)
    else:
        raise NotImplementedError('{} is not supported action-type!'.format(c.action_type))
    
    # Image
    if ('vit' in c.enc_arch) or ('efficient' in c.enc_arch):
        encoder = timm.create_model(c.enc_arch, pretrained=True)
        arch_config = resolve_data_config({}, model=encoder)
        c.norm_mean, c.norm_std = list(arch_config['mean']), list(arch_config['mean'])
        c.img_size = arch_config['input_size'][1:]  # HxW format
        c.crp_size = arch_config['input_size'][1:]  # HxW format
    else:
        c.img_size = (c.input_size, c.input_size)  # HxW format
        c.crp_size = (c.input_size, c.input_size)  # HxW format
        if c.dataset == 'stc':
            c.norm_mean, c.norm_std = 3*[0.5], 3*[0.225]
        else:
            c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    c.img_dims = [3] + list(c.img_size)
    # Network hyperparameters
    c.clamp_alpha = 1.9  # see paper equation 2 for explanation
    c.condition_vec = 128
    c.dropout = 0.0  # dropout in s-t-networks

    # Dataloader parameters
    if c.ood_noise_type == '':
        # data_folder_path = 'data' # eeee
        data_folder_path = '/home/tri/MultilevelAD'
        
        if c.dataset in ['mvtec', 'VisA']:
            dataset_category = 'Industry'
        elif c.dataset in ['example']:
            dataset_category = 'NoveltyClass'
        elif c.dataset in ['covid19', 'diabetic_retinopathy', 'skin_lesion']:
            dataset_category = 'Medical'
        else:
            raise NotImplementedError('Dataset {} is not supported!'.format(c.dataset))

        if c.dataset in ['mvtec', 'VisA', 'covid19', 'diabetic_retinopathy', 'skin_lesion']:
            c.data_path = os.path.join(data_folder_path, dataset_category, c.dataset.lower().replace('_', '-'))
        elif c.dataset == 'example':
            c.data_path = os.path.join(data_folder_path, dataset_category, 'multidog')
        else: assert False, 'Dataset {} is not supported!'.format(c.dataset)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    
    # Output settings
    c.verbose = True
    c.hide_tqdm_bar = True
    c.save_results = True
    # Unsup-train
    c.print_freq = 2
    c.temp = 0.5
    c.lr_decay_epochs = [i*c.meta_epochs//100 for i in [50,75,90]]
    print('LR schedule: {}'.format(c.lr_decay_epochs))
    c.lr_decay_rate = 0.1
    c.lr_warm_epochs = 2
    c.lr_warm = True
    c.lr_cosine = True
    if c.lr_warm:
        c.lr_warmup_from = c.lr/10.0
        if c.lr_cosine:
            eta_min = c.lr * (c.lr_decay_rate ** 3)
            c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                    1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)) / 2
        else:
            c.lr_warmup_to = c.lr

    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    c.device = torch.device("cuda" if c.use_cuda else "cpu")
    if c.action_type in ['norm-train', 'norm-test']:
        if c.action_type == 'norm-train':
            from train import train
        else:
            from train_modify_get_logit_scores import train
        train(c)
    else:
        raise NotImplementedError('{} is not supported action-type!'.format(c.action_type))


if __name__ == '__main__':
    c = get_args()
    main(c)
