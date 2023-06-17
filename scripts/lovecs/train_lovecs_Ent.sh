#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='LoveDA.st.lovecs.2urban'
python ./methods/loveda/LoveCS_Ent_train.py --config_path=${config_path}