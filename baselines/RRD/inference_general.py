import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj, evaluation_ours_class
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import MVTecDataset_test, get_data_transforms, ClassDataset_test, GeneralDataset_test


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inference(dataset_name, subset, test_path_data, root):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(256, 256)
    

    checkpoint_class  = f'./checkpoint/{dataset_name}/wres50_{subset}.pth'
    
    print(checkpoint_class )
    test_data = GeneralDataset_test(transform=data_transform, test_path_data=test_path_data, root=root)
    print(len(test_data))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Use pretrained wide_resnet50 for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj_layer =  MultiProjectionLayer(base=64).to(device)

    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
  
    pr_list_max,pr_list_sum,paths,levels = evaluation_ours_class(encoder, proj_layer, bn, decoder, test_dataloader, device)        
    
    return pr_list_max,pr_list_sum,paths,levels


if __name__ == '__main__':
    # pars = get_args()
    datasets = ['multidog', 'mvtec', 'visa', 'diabetic-retinopathy', 'covid19', 'skin-lesion']

    
    subset_dict = {
    'multidog': ['bichon_frise','chinese_rural_dog','golden_retriever','labrador_retriever','teddy'],
    'mvtec': ['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','transistor','zipper'],
    'visa': ['capsules', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pipe_fryum'],
    'diabetic-retinopathy': ['diabetic-retinopathy'],
    'covid19': ['covid19'],
    'skin-lesion': ['skin-lesion']
    }
    

    for dataset_name in datasets:
        if dataset_name == 'covid':
            root_general = '.../data/Medical/covid19/'
        elif dataset_name == 'diabetic':
            root_general = '.../data/Medical/diabetic-retinopathy/'
        elif dataset_name == 'multidog':
            root_general = '.../data/class-based/multidog/'
            
        elif dataset_name == 'mvtec':
            root_general = '.../data/Industry/mvtec_order/'
            
            
            
        elif dataset_name == 'visa':
            root_general = '.../data/Industry/visa/'
        elif dataset_name == 'skin-lesion':
            root_general = '.../data/Medical/skin-lesion/'
            
            
                    
        if dataset_name != 'diabetic' and dataset_name != 'covid' and dataset_name != 'skin-lesion':
            subsets = subset_dict[dataset_name]
                
            for subset_name in subsets:
                path_test_data = '.../data/template/' + dataset_name + '_' + subset_name + '_template.csv' 
                if dataset_name != 'multidog':
                    root = root_general + subset_name + '/test/'
                else:
                    root = root_general
                    
                
                pr_list_max,pr_list_sum,paths,levels = inference(test_path_data=path_test_data, subset=subset_name, dataset_name=dataset_name, root=root)
                paths = [item.replace(root,'') for item in paths]
                df = pd.DataFrame({
                    'Path': paths,
                    'Severity': levels,
                    'Anomaly Score': pr_list_max,
                })

                df.to_csv('.../results/RRD/RRD_' + dataset_name + '_' + subset_name +'.csv', index=False)
        else:
            subset_name = dataset_name
            path_test_data = '.../data/template/' + dataset_name + '_template.csv' 
            root = root_general
                
            pr_list_max,pr_list_sum,paths,levels = inference(test_path_data=path_test_data, subset=subset_name, dataset_name=dataset_name, root=root)
            paths = [item.replace(root,'') for item in paths]
            df = pd.DataFrame({
                'Path': paths,
                'Severity': levels,
                'Anomaly Score': pr_list_max,
            })

            df.to_csv('.../results/RRD/RRD_' + dataset_name +'.csv', index=False)
            
        