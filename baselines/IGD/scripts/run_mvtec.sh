#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m p256.ssim_main --num 1 2 3 4 5 6 7 8 9 10 11 12 14 15 --sample_rate 1.0 --work_dir "$(pwd)"