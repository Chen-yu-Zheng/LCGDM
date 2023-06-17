#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='st.iast.2urban'
python methods/loveda/IAST_Ent_train.py --config_path=${config_path}