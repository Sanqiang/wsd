#!/usr/bin/env bash

# For extra def
CUDA_VISIBLE_DEVICES=0 nohup python -m model.test -testset share -env luoz3 -ngpus 1 -mode base -out 1022_extradef_test_share_2 -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr --extra_loss def>1022_extradef_test_share.log &
CUDA_VISIBLE_DEVICES=9 nohup python -m model.test -testset msh -env luoz3 -ngpus 1 -mode base -out 1022_extradef_test_msh_2 -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr --extra_loss def>1022_extradef_test_msh.log &
