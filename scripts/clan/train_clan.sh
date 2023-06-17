#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='adv.clan.2urban'
python CLAN_train.py --config_path=${config_path}
