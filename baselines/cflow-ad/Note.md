*** Commands ***
sample.sh


*** Options ***
export CUDA_VISIBLE_DEVICES=1


*** Notation ***
Csv files is stored in the csv_results folder. File csv include anomaly score is name as '*_s.csv'

+ When you want to export the anomaly score for new dataset:
- Change the input_file and output_file in train_modify_get_logit_scores.py to your csv file path.
+ When you want to train the model, do the inverse of the above.