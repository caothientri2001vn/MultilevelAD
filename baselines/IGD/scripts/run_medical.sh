#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m p256.ssim_main_medical --num 1 2 3 --sample_rate 1.0 --work_dir "$(pwd)"