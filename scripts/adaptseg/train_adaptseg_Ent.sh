#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='adv.adaptseg.2urban'
python methods/loveda/AdaptSegNet_Ent_train.py --config_path=${config_path}