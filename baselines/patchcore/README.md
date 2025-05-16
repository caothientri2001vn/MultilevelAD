## Environment
Create and activate the conda environment using the provided `environment.yml` file. If you need to change the environment name, check the `name` field in the `environment.yml` file.

```bash
conda env create -f environment.yml
```

## Train and Evaluate

You must move the `data` folder to this directory before running the shell scripts (mv data baselines/patchcore-inspection/).

We provided shell scripts to train and evaluate the models on our dataset. To train and evaluate with our default settings, run the corresponding shell script in the `scripts.sh` file.

You may want to change the GPU device or the hyper-parameters. Please check the shell scripts for more details.

## Acknowledgement
This implementation is adapted and modified based on the original [patchcore-inspection](https://github.com/amazon-science/patchcore-inspection) code. We are thankful to their brilliant works!

