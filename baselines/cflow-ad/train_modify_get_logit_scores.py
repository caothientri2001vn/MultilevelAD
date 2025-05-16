import os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *

import csv

OUT_DIR = './viz/'

gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()


def test_meta_epoch(c, epoch, loader, encoder, decoders, pool_layers, N):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    start = time.time()
    with torch.no_grad():
        for i, (image, label, mask, image_idx) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # save
            print('image_idx', image_idx)
            assert all(image_idx[i] < image_idx[i+1] for i in range(len(image_idx)-1))
            if c.viz:
                image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer]  # BxCxHxW

                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    m_p = m_r[idx] > 0.5  # Nx1
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
    #
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))
    #
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list


def train(c):
    print('Args:', c)
    
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    L = c.pool_layers # number of pooled layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()

    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims] # this block is time consuming
        
    decoders = [decoder.to(c.device) for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())
    optimizer = torch.optim.Adam(params, lr=c.lr)
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    
    # Data
    if c.dataset in ['mvtec', 'VisA', 'example', 'covid19', 'diabetic_retinopathy', 'skin_lesion']:
        test_dataset  = MVTecDataset(c, is_train=False)
    elif c.dataset == 'stc':
        train_dataset = StcDataset(c, is_train=True)
        test_dataset  = StcDataset(c, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
    print('Dataset', len(test_loader.dataset))
    print('Dataloader', len(test_loader))
    # Stats
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_pro_obs = Score_Observer('SEG_AUPRO')
    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs):
        assert c.action_type == 'norm-test' and c.checkpoint
        load_weights(encoder, decoders, c.checkpoint)
        print('Done loading weights')
        
        height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(
            c, epoch, test_loader, encoder, decoders, pool_layers, N)

        test_map = [list() for p in pool_layers]
        for l, p in enumerate(pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[l], width[l])
            test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
        score_map = np.zeros_like(test_map[0])
        for l, p in enumerate(pool_layers):
            score_map += test_map[l]
        score_mask = score_map # score_mask (83, 512, 512) float64
        super_mask = score_mask.max() - score_mask # super_mask (83, 512, 512) float64
        score_label = np.max(super_mask, axis=(1, 2)) # score_label (83,) float64
        print('score_label', score_label.shape, score_label.dtype)
        gt_label = np.asarray(gt_label_list, dtype=bool) # gt_label (83,) bool
        
        
        ### This is my custom code to store the anomaly score to csv file's
        _classname = c.class_name
        assert len(test_loader.dataset) == len(score_label)
        csv_template_folder = './data/template'
        
        if 'visa' == c.dataset.lower():
            input_file = os.path.join(csv_template_folder, f'{c.dataset.lower()}_{_classname}_template.csv')
        elif 'example' == c.dataset.lower():
            input_file = os.path.join(csv_template_folder, f'multidog_{_classname}_template.csv')
        elif 'diabetic_retinopathy' == c.dataset.lower():
            input_file = os.path.join(csv_template_folder, 'diabetic-retinopathy_template.csv')
        elif 'covid19' == c.dataset.lower():
            input_file = os.path.join(csv_template_folder, f'covid19_template.csv')
        elif 'mvtec' == c.dataset.lower():
            input_file = os.path.join(csv_template_folder, f'{c.dataset.lower()}_{_classname}_template.csv')
        elif 'skin_lesion' in c.dataset.lower():
            input_file = os.path.join(csv_template_folder, 'skin-lesion_template.csv')
        else: raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
        output_file = input_file.replace('.csv', '_s.csv')
            
        print('input_file', input_file)
        print('output_file', output_file)
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            header = next(reader)
            header = header[:2]
            header.append('Anomaly Score')
            writer.writerow(header)
        
            for score in score_label:
                row = next(reader)
                row = row[:2]
                row.append(score)
                writer.writerow(row)

            print(f"New CSV file with scores has been created: {output_file}")
        print('Done!')
        