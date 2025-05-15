import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
import base64

from openai import OpenAI
from anthropic import Anthropic
import glob

import re
class AnomalyDetector:
    def __init__(self, model_name):
        if "gpt" in model_name:
            self.client = OpenAI(api_key="Specify your OpenAI API key here")
        elif "claude" in model_name:
            self.client = Anthropic(api_key="Specify your Anthropic API key here")
            
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

        return plain_text
        
    def set_seed(self, seed=42):
        np.random.seed(seed)

    def load_dataset(self, test_path_data, root):
        df = pd.read_csv(test_path_data)
        columns_as_lists = {col: df[col].tolist() for col in df.columns}
        img_tot_paths = columns_as_lists['Path']
        img_tot_paths = [root + item for item in img_tot_paths]
        tot_types = columns_as_lists['Severity']


        return img_tot_paths, tot_types
    
    def extract_anomaly_details(self, text):
        anomaly_score_pattern = r'Anomaly Score: (\d+)'
        reason_pattern = r'Reason: (.+)'

        anomaly_score_match = re.search(anomaly_score_pattern, text)
        reason_match = re.search(reason_pattern, text)

        if anomaly_score_match and reason_match:
            anomaly_score = int(anomaly_score_match.group(1))
            reason = reason_match.group(1)
            return anomaly_score, reason
        else:
            return 0, None
        
    def inference(self, image_path_test, normal_image_paths, subset,model_name, context=None):
        base64_image_test = self.encode_image(image_path_test)
        base64_image_normal1 = self.encode_image(normal_image_paths[0])
        base64_image_normal2 = self.encode_image(normal_image_paths[1])
        base64_image_normal3 = self.encode_image(normal_image_paths[2])
        
        
        
        prompt = f"""
You are given four images:

Image 1: Belongs to the normal class {subset}.
Image 2: Also belongs to the normal class {subset}.
Image 3: Also belongs to the normal class {subset}.
Image 4: The inference image to be analyzed.

Your task is to determine an anomaly score for the inference image. This score should indicate how similar the inference image is to the class of the normal images. A lower anomaly score means the inference image is similar to the normal class, while a higher score indicates a greater deviation.
Take into account the relationship between the class of the normal images and the class of the inference image. Carefully consider that there may be different levels of relationships between the normal class and the various abnormal classes. For example, some abnormal classes might be more closely related to the normal class (e.g., two different breeds of cats), while others might be completely different (e.g., a cat versus a completely unrelated object). These relationships should reflect in the anomaly score:
- If the inference image belongs exactly to the same class as the normal images, assign a score of 0.
- If the inference image is similar but not identical to the normal class, assign a score between 1 and 50, with lower values indicating greater similarity.
- If the inference image belongs to an abnormal class but has some level of relationship to the normal class, assign a score between 51 and 75, considering the degree of similarity.
- If the inference image is completely unrelated to the normal class, assign a score between 76 and 100, with higher values indicating a greater dissimilarity.

Output format: 
Anomaly Score: [Score] 
Reason: [Provide a brief explanation of why this score was assigned]

Please ensure the output strictly follows the format. You must answer the question even you are not sure about the anwser."""
        if "gpt" in model_name: 
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that responds in detecting anomalies on images in various context."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Normal Image 1:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image_normal1}"}
                            },
                            {"type": "text", "text": "Normal Image 2:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image_normal2}"}
                            },
                            {"type": "text", "text": "Normal Image 3:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image_normal3}"}
                            },
                            {"type": "text", "text": "Inference Image :"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image_test}"}
                            }       
                        ]}
                    ],
                    
                    temperature=0.0,
                    max_tokens=300,
                )
                outputs = response.choices[0].message.content
            except Exception as e:
                print(e)
                outputs = "Error"
        elif "claude" in model_name:
            try:
                response = self.client.messages.create(
                    model=model_name,
                    system="You are a helpful assistant that responds in detecting anomalies on images in various context.",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Normal Image 1:"},
                            {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image_normal1,}
                            },
                            {"type": "text", "text": "Normal Image 2:"},
                            {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image_normal2,}
                            },
                            {"type": "text", "text": "Normal Image 3:"},
                            {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image_normal3,}
                            },
                            {"type": "text", "text": "Inference Image :"},
                            {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image_test,}
                            },     
                        ]}
                    ],
                    
                    temperature=0.0,
                    max_tokens=300,
                )
                outputs = response.content[0].text
            except Exception as e:
                outputs = "Error"
            
        anomaly_score, reason = self.extract_anomaly_details(outputs)

        return anomaly_score, reason
        
        
    def run_batch(self, model_name, results_path, test_path_data, normal_image_paths, subset, root,context=None):
        self.set_seed()

        # results file loading
        need_to_create_new_results_file = True
        if os.path.exists(results_path):
            try:
                df_results = pd.read_json(results_path, orient="index")
                need_to_create_new_results_file = False
                print("Results file loaded.")
            except:
                need_to_create_new_results_file = True
                print("Results file not found or corrupted. Will create a new one.")
        if need_to_create_new_results_file:
            df_results = pd.DataFrame(data={
                    "Path": [],
                    "Severity": [],
                    "Anomaly Score": [],
                    "Reason": [],
                    })
            print("Results file created.")
        dones = set(df_results['Path'].tolist())


        img_tot_paths, tot_types = self.load_dataset(test_path_data=test_path_data, root=root)
        image_paths = []
        anomaly_scores = []
        severities = []

        for index in tqdm(range(len(img_tot_paths))):
        # for index in tqdm(range(10)):
            test_image_path = img_tot_paths[index]
            if test_image_path.replace(root,'') in dones:
                continue
            severity = tot_types[index]
            anomaly_score, reason = self.inference(image_path_test=test_image_path, normal_image_paths=normal_image_paths, subset=subset, model_name=model_name)
            
            image_paths.append(test_image_path.replace(root,''))
            anomaly_scores.append(anomaly_score)
            severities.append(severity)
            try:
                df_results.loc[len(df_results)] = {
                    "Path": test_image_path.replace(root,''),
                    "Severity": severity,
                    "Anomaly Score": anomaly_score,
                    "Reason": reason
                }
                df_results.to_json(results_path, orient="index", indent=4)
            except UnicodeEncodeError:
                continue
        return image_paths, severities, anomaly_scores
            
                



model_name = "gpt-4o" #"gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"
dataset_name = 'multidog'
ADetector = AnomalyDetector(model_name)

subsets = ['bichon_frise','chinese_rural_dog','golden_retriever','labrador_retriever','teddy']

root_general = '../data/OneClassNovelty/multidog/'


for subset_name in subsets:
    path_test_data = '../data/template/' + dataset_name + '_' + subset_name + '_template.csv' 

    normal_image_files = glob.glob(os.path.join(root_general + '/level_0_train/' + subset_name, "*.*"))
    sorted_images = sorted(normal_image_files)
    normal_image_paths = sorted_images[0:3]

    
    results_path = f"./results_json/{model_name}_{dataset_name}_{subset_name}.json"
    
    
    paths,levels,anomalies = ADetector.run_batch(test_path_data=path_test_data, 
    subset=subset_name, root=root_general, model_name=model_name, results_path=results_path,
    normal_image_paths=normal_image_paths)
    paths = [item.replace(root_general,'') for item in paths]
    df = pd.DataFrame({
        'Path': paths,
        'Area': levels,
        'Anomaly Score': anomalies,
    })


    # Save the DataFrame to a CSV file
    df.to_csv(f'../results/{model_name}/{model_name}_{dataset_name}_{subset_name}.csv', index=False)