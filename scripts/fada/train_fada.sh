#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='adv.fada.2urban'
python methods/loveda/FADA_train.py --config_path=${config_path}
