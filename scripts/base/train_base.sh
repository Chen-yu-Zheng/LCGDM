#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='base.2urban'
python methods/loveda/Baseline_train.py --config_path=${config_path}
