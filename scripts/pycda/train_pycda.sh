#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='st.pycda.2urban'
python methods/loveda/PyCDA_train.py --config_path=${config_path}
