import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
import geomloss
from fastprogress import progress_bar
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from dataset.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms, ClassDataset_test, ClassDataset_train

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save_folder', default = './checkpoint/multidog', type=str)
    parser.add_argument('--batch_size', default = 64, type=int)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--num_epoch', default = 100, type=int)
    parser.add_argument('--num_sample', default = 500, type=int)
    parser.add_argument('--detail_training', default='note', type = str)
    parser.add_argument('--proj_lr', default = 0.001, type=float)
    parser.add_argument('--distill_lr', default = 0.005, type=float)
    parser.add_argument('--weight_proj', default = 0.2, type=float) 
    parser.add_argument('--classes', nargs="+", default=['bichon_frise','chinese_rural_dog','golden_retriever','labrador_retriever','teddy'])
    
    pars = parser.parse_args()
    return pars

def train(_class_, pars):
    print(_class_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    train_path = '.../data/class-based/OneClassNovelty/level_0_train/'+_class_
    
    save_model_path  = pars.save_folder + '/' + 'wres50_'+_class_+'.pth'
    train_data = ClassDataset_train(root=train_path, transform=data_transform, num_sample=pars.num_sample)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=pars.batch_size, shuffle=True)

    # Use pretrained ImageNet for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    proj_loss = Revisit_RDLoss()
    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=pars.proj_lr, betas=(0.5,0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=pars.distill_lr, betas=(0.5,0.999))


    num_epoch = pars.num_epoch

    print(f'with class {_class_}, Training with {num_epoch} Epoch')
    
    for epoch in tqdm(range(1,num_epoch+1)):
        bn.train()
        proj_layer.train()
        decoder.train()
        loss_proj_running = 0
        loss_distill_running = 0
        total_loss_running = 0
        
        ## gradient acc
        accumulation_steps = 2
        
        for i, (img,img_noise,_) in enumerate(train_dataloader):
            img = img.to(device)
            img_noise = img_noise.to(device)
            inputs = encoder(img)
            inputs_noise = encoder(img_noise)

            (feature_space_noise, feature_space) = proj_layer(inputs, features_noise = inputs_noise)

            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)

            outputs = decoder(bn(feature_space))#bn(inputs))
            L_distill = loss_fucntion(inputs, outputs)
            loss = L_distill + pars.weight_proj * L_proj
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer_proj.step()
                optimizer_distill.step()
                # Clear gradients
                optimizer_proj.zero_grad()
                optimizer_distill.zero_grad()
            
            total_loss_running += loss.detach().cpu().item()
            loss_proj_running += L_proj.detach().cpu().item()
            loss_distill_running += L_distill.detach().cpu().item()
        print("Epoch: ", epoch)
        print(total_loss_running, loss_proj_running, loss_distill_running)

        if epoch%50 == 0 or epoch == 1: 
            torch.save({'proj': proj_layer.state_dict(),
                        'decoder': decoder.state_dict(),
                        'bn':bn.state_dict()}, save_model_path)





if __name__ == '__main__':
    pars = get_args()
    print('Training with classes: ', pars.classes)

    setup_seed(111)
    for c in pars.classes:
        train(c, pars)
