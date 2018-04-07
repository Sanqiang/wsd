#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/home/zhaos5/projs/wsd/wsd_code"

CUDA_VISIBLE_DEVICES=0 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode selfattn>base_train.log &
CUDA_VISIBLE_DEVICES=1 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base_avg -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode avg>base_train_avg.log &
CUDA_VISIBLE_DEVICES=2 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base_notime -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode selfattn -pos none>base_train_notime.log &
CUDA_VISIBLE_DEVICES=3 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base_padmask -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode selfattn -cprocess padmask >base_train_padmask.log &


CUDA_VISIBLE_DEVICES=10 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode selfattn>base_eval.log &
CUDA_VISIBLE_DEVICES=11 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base_avg -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode avg>base_eval_avg.log &
CUDA_VISIBLE_DEVICES=12 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base_notime -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode selfattn -pos none>base_train_notime.log &
CUDA_VISIBLE_DEVICES=13 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base_padmask -max_context_len 1000 -dim 512 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 8 -ag_mode selfattn -cprocess padmask>base_train_padmask.log &


CUDA_VISIBLE_DEVICES=4 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 5 -ag_mode selfattn>base_train_small.log &
CUDA_VISIBLE_DEVICES=5 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base_avg_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 5 -ag_mode avg>base_train_avg_small.log &
CUDA_VISIBLE_DEVICES=6 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base_notime_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 5 -ag_mode selfattn -pos none>base_train_notime_small.log &
CUDA_VISIBLE_DEVICES=7 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out base_padmask_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 6 -nel 6 -ndl 6 -nh 5 -ag_mode selfattn -cprocess padmask >base_train_padmask_small.log &


CUDA_VISIBLE_DEVICES=8 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 4 -nel 4 -ndl 4 -nh 5 -ag_mode selfattn>base_eval_small.log &
CUDA_VISIBLE_DEVICES=9 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base_avg_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 4 -nel 4 -ndl 4 -nh 5 -ag_mode avg>base_eval_avg_small.log &
CUDA_VISIBLE_DEVICES=10 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base_notime_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 4 -nel 4 -ndl 4 -nh 5 -ag_mode selfattn -pos none>base_train_notime_small.log &
CUDA_VISIBLE_DEVICES=11 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out base_padmask_small -max_context_len 2000 -dim 300 -bsize 6 -nhl 4 -nel 4 -ndl 4 -nh 5 -ag_mode selfattn -cprocess padmask>base_train_padmask_small.log &
