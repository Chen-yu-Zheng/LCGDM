#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='st.iast.2urban'
python IAST_SL_train.py --config_path=${config_path}