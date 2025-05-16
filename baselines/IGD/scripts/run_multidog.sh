#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m p256.ssim_main_multidog --num 1 2 3 4 5 --sample_rate 1.0 --work_dir "$(pwd)"