#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='LoveDA.st.lovecs.2urban'
python ./methods/loveda/LoveCS_train.py --config_path=${config_path}