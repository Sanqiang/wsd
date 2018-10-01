#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/home/zhaos5/projs/wsd/wsd_code"

# For base
CUDA_VISIBLE_DEVICES=1 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train -pred_mode match -max_context_len 500 -dim 256 -bsize 32 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode sense>0930_base_abbrsense_train.log &
CUDA_VISIBLE_DEVICES=15 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode sense>0930_base_abbrsense_eval.log &

CUDA_VISIBLE_DEVICES=2 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out 0930_base_abbrabbr_train -pred_mode match -max_context_len 500 -dim 256 -bsize 32 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode abbr>0930_base_abbrabbr_train.log &
CUDA_VISIBLE_DEVICES=14 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_base_abbrabbr_train -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr>0930_base_abbrabbr_eval.log &

# For extra
CUDA_VISIBLE_DEVICES=3 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train_extra -pred_mode match -max_context_len 500 -dim 256 -bsize 32 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode sense --extra_loss def:stype>0930_base_abbrsense_extra_train.log &
CUDA_VISIBLE_DEVICES=13 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train_extra -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode sense --extra_loss def:stype>0930_base_abbrsense_extra_eval.log &

CUDA_VISIBLE_DEVICES=4 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out 0930_base_abbrabbr_train_extra -pred_mode match -max_context_len 500 -dim 256 -bsize 32 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode abbr --extra_loss def:stype>0930_base_abbrabbr_extra_train.log &
CUDA_VISIBLE_DEVICES=12 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_base_abbrabbr_train_extra -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr --extra_loss def:stype>0930_base_abbrabbr_extra_eval.log &

# For extra def
CUDA_VISIBLE_DEVICES=5 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train_extradef -pred_mode match -max_context_len 500 -dim 256 -bsize 32 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode sense --extra_loss def>0930_base_abbrsense_extradef_train.log &
CUDA_VISIBLE_DEVICES=11 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train_extradef -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode sense --extra_loss def>0930_base_abbrsense_extradef_eval.log &

CUDA_VISIBLE_DEVICES=6 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out 0930_base_abbrabbr_train_extradef -pred_mode match -max_context_len 500 -dim 256 -bsize 32 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode abbr --extra_loss def>0930_base_abbrabbr_extradef_train.log &
CUDA_VISIBLE_DEVICES=10 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_base_abbrabbr_train_extradef -pred_mode match -max_context_len 500 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode abbr --extra_loss def>0930_base_abbrabbr_extradef_eval.log &

# for extra big
CUDA_VISIBLE_DEVICES=7 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train_extra_big -pred_mode match -max_context_len 500 -dim 512 -bsize 32 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode sense --extra_loss def:stype>0930_base_abbrsense_extra_big_train.log &
CUDA_VISIBLE_DEVICES=8 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out 0930_base_abbrsense_train_extra_big -pred_mode match -max_context_len 500 -dim 512 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg --abbr_mode sense --extra_loss def:stype>0930_base_abbrsense_extra_big_eval.log &

