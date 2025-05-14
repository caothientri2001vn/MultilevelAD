import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import base64
from openai import OpenAI
from anthropic import Anthropic
import glob
import re
from PIL import Image
from io import BytesIO
class AnomalyDetector:
    def __init__(self, model_name):
        if "gpt" in model_name:
            self.client = OpenAI(api_key="Specify your OpenAI API key here")
        elif "claude" in model_name:
            self.client = Anthropic(api_key="Specify your Anthropic API key here")
            
    
    def encode_image_resize(self, image_path):
        image = Image.open(image_path)
        if image.height > 512:
            new_size = (int(image.width * (512/image.height)), 512)
            image = image.resize(new_size, Image.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return image_base64
        
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
        
    def get_promt(self, dataset):
        if dataset == "skin-lesion":
            prompt = """
You are given four dermoscopic images:

Image 1, 2, 3: Belong to the normal skin category.
Image 4: The inference image to be analyzed.

Your task is to determine an anomaly score for the inference image, reflecting its potential danger level in comparison to normal skin. A lower anomaly score suggests that the lesion is benign, while a higher score indicates a more dangerous lesion type, such as precancerous or cancerous.

Use the following guidelines and ranges to assign the anomaly score:
- If the inference image is consistent with normal skin, assign a score of 0.
- If the inference image appears benign but shows minor atypical features, such as a mole (nevus) with slightly irregular borders, assign a score between 1 and 30 to indicate a benign lesion.
- If the inference image has characteristics suggesting it could be precancerous, assign a score between 31 and 70.
- If the inference image displays features strongly associated with malignancy, assign a score between 71 and 100, indicating a high risk for cancer.
Output format:
Anomaly Score: [Score]
Reason: [Provide a brief explanation of why this score was assigned]

Please ensure the output strictly follows the format. You must answer the question even if you are not entirely certain about the answer. Do not answer that you can not."""

        elif dataset == "covid":
            prompt = """
You are given four lung scan images:

Image 1, 2, 3: Belongs to the normal lung category. 
Image 4: The inference image to be analyzed.

Your task is to determine an anomaly score for the inference image on a scale from 0 to 100, reflecting the extent of lung damage. A lower anomaly score suggests minimal or no lung damage, while a higher score indicates more severe lung damage.

Use the following guidelines to assign the anomaly score:
- If the inference image is consistent with a normal lung, assign a score of 0.
- If the inference image shows very mild lung damage, such as isolated ground-glass opacities, assign a score between 1 and 15.
- If the inference image shows mild lung damage, with limited areas of ground-glass opacities, assign a score between 16 and 30.
- If the inference image has moderate lung damage, such as more widespread ground-glass opacities, assign a score between 31 and 50.
- If the inference image displays moderately severe lung damage, including partial lung consolidation, assign a score between 51 and 70.
- If the inference image shows severe lung damage, such as extensive consolidation in multiple lung regions, assign a score between 71 and 85.
- If the inference image reveals extreme lung damage, affecting more than 85 percent of the lungs, assign a score between 86 and 100.
Output format:
Anomaly Score: [Score]
Reason: [Provide a brief explanation of why this score was assigned]

Please ensure the output strictly follows the format. You must answer the question even if you are not entirely certain about the answer. Do not answer that you can not."""

        elif dataset == "diabetic":
            prompt = """
You are given four retinal images:

Image 1, 2, 3: Belongs to the normal retinal category (No DR). 
Image 4: The inference image to be analyzed.

Your task is to determine an Anomaly Score for the inference image on a scale from 0 to 100, reflecting the extent of diabetic retinopathy (DR). A lower anomaly score suggests minimal or no DR, while a higher score indicates more severe stages of DR.

Use the following guidelines to assign the Anomaly Score:
- If the inference image shows no signs of diabetic retinopathy, assign a score of 0.
- If the inference image shows mild signs of diabetic retinopathy, assign an anomaly score between 1 and 25.
- If the inference image has characteristics suggesting moderate diabetic retinopathy, assign an anomaly score between 26 and 50.
- If the inference image displays severe diabetic retinopathy, assign an anomaly score between 51 and 75.
- If the inference image reveals advanced proliferative diabetic retinopathy, assign an anomaly score between 76 and 100.
Output format:
Anomaly Score: [Score]
Reason: [Provide a brief explanation of why this score was assigned]

Please ensure the output strictly follows the format. You must answer the question even if you are not entirely certain about the answer. Do not answer that you can not."""

        return prompt
        
    def inference(self, image_path_test, normal_image_paths, subset,model_name, context=None):
        base64_image_test = self.encode_image_resize(image_path_test)
        base64_image_normal1 = self.encode_image_resize(normal_image_paths[0])
        base64_image_normal2 = self.encode_image_resize(normal_image_paths[1])
        base64_image_normal3 = self.encode_image_resize(normal_image_paths[2])
        
        
        
        prompt = self.get_promt(subset)
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
                                    "media_type": f"image/png",
                                    "data": base64_image_normal1,}
                            },
                            {"type": "text", "text": "Normal Image 2:"},
                            {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": f"image/png",
                                    "data": base64_image_normal2,}
                            },
                            {"type": "text", "text": "Normal Image 3:"},
                            {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image_normal3,}
                            },
                            {"type": "text", "text": "Inference Image :"},
                            {"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": f"image/png",
                                    "data": base64_image_test,}
                            },     
                        ]}
                    ],
                    
                    temperature=0.0,
                    max_tokens=300,
                )
                outputs = response.content[0].text
                # print()
            except Exception as e:
                print(e)
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
            

datasets = ['diabetic-retinopathy', 'covid19', 'skin-lesion']
model_name = "gpt-4o" # "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"
ADetector = AnomalyDetector(model_name)

for dataset_name in datasets:
    root_general = f'../data/Medical/{dataset_name}'    
                
    subset_name = dataset_name
    path_test_data = f'../data/template/{dataset_name}_template.csv' 
    root = root_general
    normal_image_files = glob.glob(os.path.join(root_general + 'level_0_train/', "*.*"))
    
    sorted_images = sorted(normal_image_files)
    normal_image_paths = sorted_images[0:3]
    
    results_path = f"./results_json/{model_name}_{dataset_name}.json"
    
    paths,levels,anomalies = ADetector.run_batch(test_path_data=path_test_data, 
    subset=subset_name, root=root, model_name=model_name, results_path=results_path,
    normal_image_paths=normal_image_paths)
    paths = [item.replace(root,'') for item in paths]
    df = pd.DataFrame({
        'Path': paths,
        'Severity': levels,
        'Anomaly Score': anomalies,
    })

    # Save the DataFrame to a CSV file
    df.to_csv(f'../results/{model_name}/{model_name}_{dataset_name}.csv', index=False)
        