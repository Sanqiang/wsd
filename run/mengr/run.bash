#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/home/mengr/project/wsd/wsd_code"

# For base
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python ../../model/train.py -env aws_mengr -arch abbr_residual -ngpus 4 -mode base -out 20181021_base_abbrabbr -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode abbr > 20181021_base_abbrabbr_train.log &
CUDA_VISIBLE_DEVICES=5 nohup python ../../model/eval.py -env aws_mengr -arch abbr_residual -ngpus 4 -mode base -out 20181021_base_abbrabbr -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr > 20181021_base_abbrabbr_eval.log &
