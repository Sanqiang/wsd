#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/home/zhaos5/projs/wsd/wsd_code"
nohup python ../model/train.py -env aws -mode base -out base -dim 512 -bsize 64 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode selfattn -max_context_len 1200>base_train.log
