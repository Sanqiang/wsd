#!/usr/bin/env bash

#export PYTHONPATH="${PYTHONPATH}:/home/zhaos5/projs/wsd/wsd_code"
export PYTHONPATH="${PYTHONPATH}:/home/luoz3/wsd_new_2/wsd"

# For extra def
sudo CUDA_VISIBLE_DEVICES=0 nohup python ../../model/test.py -env aws -ngpus 1 -mode base -out 0930_extradef_test_share -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr --extra_loss def>0930_extradef_test_share.log &
#CUDA_VISIBLE_DEVICES=9 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_extradef_test_msh -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr --extra_loss def>0930_extradef_test_msh.log &
