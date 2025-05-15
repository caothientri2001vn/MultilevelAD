import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau
from lifelines.utils import concordance_index


def make_per(result_path):
    data = pd.read_csv(result_path)

    severity = data['Severity']

    anomaly_scores = data['Anomaly Score']

    # Ensure Severity is treated as numeric
    severity = pd.to_numeric(severity, errors='coerce').apply(lambda x: np.ceil(x))

    # ROCAUC Calculation: between severity 0 and each other level, and between severity 0 and the whole
    roc_auc_results = {}
    severity_levels = severity.unique()
    list_results = []
    for level in severity_levels:
        if level != 0:
            # Filtering data to calculate ROC AUC between severity 0 and each other level
            mask = (severity == 0) | (severity == level)
            binary_severity = np.where(severity[mask] == 0, 0, 1)  # Convert to binary (0 vs current level)
            # print(binary_severity)
            auc = roc_auc_score(binary_severity, anomaly_scores[mask])
            roc_auc_results[f'0 vs {level}'] = auc
            list_results.append(auc)

    # ROCAUC Calculation between severity 0 and the whole dataset
    binary_severity_whole = np.where(severity == 0, 0, 1)  # Convert to binary (0 vs all other levels)
    auc_whole = roc_auc_score(binary_severity_whole, anomaly_scores)
    roc_auc_results['0 vs Whole'] = auc_whole
    list_results.append(auc_whole)

    # Kendall's Tau calculation for the entire dataset
    kendall_tau, _ = kendalltau(severity, anomaly_scores)
    list_results.append(kendall_tau)

    # Concordance Index (C-index) calculation for the entire dataset
    c_index = concordance_index(severity, anomaly_scores)
    list_results.append(c_index)

    # Display results
    print("ROCAUC Results:")
    for key, value in roc_auc_results.items():
        print(f"{key}: {value}")

    print(f"Kendall's Tau: {kendall_tau}")
    print(f"Concordance Index (C-index): {c_index}")
    list_results =[round(item*100,2) for item in list_results]
    print(', '.join(map(str, list_results))) #for copy-paste
    
    return list_results

# model_names = ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307', 'gpt-4o', 'pni', 'skipganomaly', 'RD4AD', 'patchcore-inspection', 'cflow-ad', 'RRD', 'simplenet','score_tmp', 'ae4ad', 'ocr-gan-32px'']
# model_names = ['patchcore-inspection', 'cflow-ad', 'RRD', 'ocr-gan-32px', 'IGD', 'gpt-4o', 'claude-3-5-sonnet-20241022']
model_names = ['gpt-4o_zero', 'claude-3-5-sonnet-20241022_zero', 'gpt-4o', 'claude-3-5-sonnet-20241022']


subset_dict = {
    'multidog': ['bichon_frise','chinese_rural_dog','golden_retriever','labrador_retriever','teddy'],
    'mvtec': ['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','transistor','zipper'],
    'visa': ['capsules', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pipe_fryum'],
    'diabetic-retinopathy': ['diabetic-retinopathy'],
    'covid19': ['covid19'],
    'skin-lesion': ['skin-lesion']
}

dataset_name = 'diabetic-retinopathy' # 'multidog', 'mvtec', 'visa', 'diabetic-retinopathy', 'covid1919', 'skin-lesion'

subsets = subset_dict[dataset_name]
list_results_whole = []
for subset_name in subsets:
    tmp_previous = []

    print(subset_name)
    if dataset_name != 'diabetic-retinopathy' and dataset_name != 'covid19' and dataset_name != 'skin-lesion':
        for model_name in model_names:
            print(model_name)
            path_result = f'./results/{model_name}/{model_name}_{dataset_name}_{subset_name}.csv'
            tmp = make_per(path_result)
            print("-"*10)
    else:
        for model_name in model_names:
            print(model_name)
            path_result = f'./results/{model_name}/{model_name}_{dataset_name}.csv'
            tmp = make_per(path_result)
            print("-"*10)

    print("-"*30)
    
            
            
  

