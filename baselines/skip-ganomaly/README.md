## Environment
Create and activate the conda environment using the provided `environment.yml` file. If you need to change the environment name, check the `name` field in the `environment.yml` file.

```bash
conda env create -f environment.yml
```

## Train and Evaluate
We provided shell scripts to train and evaluate the models on our dataset. To train and evaluate with our default settings, run the corresponding shell script in the `scripts/` directory. For example, to train and evaluate Skip-GANomaly on the VisA dataset, run:

```bash
mkdir result_csvs merged_result_csvs
./scripts/train_visa.sh
./scripts/infer_visa.sh
```

You may want to change the GPU device or the hyper-parameters. Please check the shell scripts for more details.

## Acknowledgement
This implementation is adapted and modified based on the original [Skip-GANomaly](https://github.com/samet-akcay/skip-ganomaly) code. We are thankful to their brilliant works!
