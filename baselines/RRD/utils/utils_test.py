import torch
from torch.nn import functional as F
import cv2
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from skimage import measure
from statistics import mean
from scipy.ndimage import gaussian_filter
import warnings
import os
from tqdm import tqdm 

def calculate_white_area(mask):
    # Threshold the mask to binary (if not already binary)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Count white pixels (255)
    white_area = np.sum(binary_mask == 255)
    return white_area

warnings.filterwarnings('ignore')
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



def evaluation_multi_proj(encoder,proj,bn, decoder, dataloader,device):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for (img, gt, label, _, _) in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),4)

def evaluation_ours_mvtec(encoder,proj,bn, decoder, dataloader,device):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    # gt_list_px = []
    # pr_list_px = []
    # gt_list_sp = []
    # pr_list_sp = []
    # aupro_list = []
    pr_list_max = []
    pr_list_sum = []
    areas = []
    paths = []
    with torch.no_grad():
        for (img, gt, label, _, path) in dataloader:
            path = path[0]
            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            ground_truth_file_path = path.replace("test","ground_truth")
            ground_truth_file_path = ground_truth_file_path.replace(".png","_mask.png")
            print(ground_truth_file_path)
            if os.path.exists(ground_truth_file_path):
                # Read the ground truth mask
                ground_truth_mask = cv2.imread(ground_truth_file_path, cv2.IMREAD_GRAYSCALE)

                # Resize the ground truth mask (assuming the test image and mask should have the same size)
                resized_mask = cv2.resize(ground_truth_mask, (256, 256))

                # Calculate the white area of the ground truth mask
                white_area = calculate_white_area(resized_mask)

                # Write the test file path and the white area to the CSV file
            elif "good" in path:
                white_area = 0

            pr_list_max.append(np.max(anomaly_map))
            pr_list_sum.append(np.sum(anomaly_map))
            paths.append(path)
            areas.append(white_area)
            print(path)
            print(white_area)
            print(np.max(anomaly_map))
            print(np.sum(anomaly_map))

    return pr_list_max,pr_list_sum,paths, areas


def evaluation_ours_class(encoder,proj,bn, decoder, dataloader,device):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    # gt_list_px = []
    # pr_list_px = []
    # gt_list_sp = []
    # pr_list_sp = []
    # aupro_list = []
    pr_list_max = []
    pr_list_sum = []
    levels = []
    paths = []
    with torch.no_grad():
        for (img, label, level, path) in tqdm(dataloader):
            path = path[0]
            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            pr_list_max.append(np.max(anomaly_map))
            pr_list_sum.append(np.sum(anomaly_map))
            levels.append(level.to('cpu').detach().numpy()[0])

            paths.append(path)
            # print(path)
            # print(np.max(anomaly_map))
            # print(np.sum(anomaly_map))

    return pr_list_max,pr_list_sum,paths,levels



def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
