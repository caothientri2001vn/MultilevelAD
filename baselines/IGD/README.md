## Train and Evaluate
We provided shell scripts to train and evaluate the models on our dataset. To train and evaluate with our default settings, run the corresponding shell script in the `scripts/` directory. For example, to train and evaluate IGD on the VisA dataset, run:

```bash
mkdir result_csvs
./scripts/run_visa.sh
```

The scripts are:

- `run_visa.sh`: Train and evaluate IGD on the VisA dataset.
- `run_mvtec.sh`: Train and evaluate IGD on the MVTec dataset.
- `run_multidog.sh`: Train and evaluate IGD on the MultiDog dataset.
- `run_medical.sh`: Train and evaluate IGD on the covid19, diabetic-retinopathy and skin-lesion datasets.

## Acknowledgement
This implementation is adapted and modified based on the original [IGD](https://github.com/tianyu0207/IGD) code. We are thankful to their brilliant works!
