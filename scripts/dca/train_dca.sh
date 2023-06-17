#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='LoveDA.st.dca.2urban'
python ./methods/loveda/DCA_train.py --config_path=${config_path}