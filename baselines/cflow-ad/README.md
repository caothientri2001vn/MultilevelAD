## Environment
Create and activate the conda environment using the provided `environment.yml` file. If you need to change the environment name, check the `name` field in the `environment.yml` file.

```bash
conda env create -f environment.yml
```

## Train and Evaluate

The path to training and testing datasets are defined in the `main.py` file. Which is the `c.data_path` variable. Simply change the `c.data_path` to your dataset path.

You must move the `data` folder to this directory before running the shell scripts (mv data baselines/patchcore-inspection/).

We provided shell scripts to train and evaluate the models on our dataset. To train and evaluate with our default settings, run the corresponding shell script in the `scripts.sh` file. For example, to train and evaluate on the VisA dataset, run:

Train:
```bash
python3 main.py --gpu 0 --pro -inp 256 --dataset covid19 --class-name covid
```

Test. Before running the test, rearrange the file in ./data/template to ../baseline/cflow-ad/result_csvs first. To do so, consider see the two variables input_file and output_file in train_modify_get_logit_scores.py.
```bash
python3 main.py --gpu 0 --pro -inp 256 --dataset covid19 --class-name covid --action-type norm-test --checkpoint PATH/FILE.PT --ood
```

You may want to change the GPU device or the hyper-parameters. Please check the shell scripts for more details.

## Acknowledgement
This implementation is adapted and modified based on the original [cflow-ad](https://github.com/gudovskiy/cflow-ad) code. We are thankful to their brilliant works!
