# Experiment for pred mode
#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/home/zhaos5/projs/wsd/wsd_code"

CUDA_VISIBLE_DEVICES=0 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out clas -pred_mode clas -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode selfattn>clas_train.log &
CUDA_VISIBLE_DEVICES=1 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out clas -pred_mode clas -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode avg>clas_eval.log &


CUDA_VISIBLE_DEVICES=2 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out match -pred_mode match -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode selfattn>match_train.log &
CUDA_VISIBLE_DEVICES=3 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match -pred_mode match -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode avg>match_eval.log &

CUDA_VISIBLE_DEVICES=4 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out match_simple -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode selfattn>match_simple_train.log &
CUDA_VISIBLE_DEVICES=5 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match_simple -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode avg>match_simple_eval.log &


CUDA_VISIBLE_DEVICES=6 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out clas_hub --hub_module_embedding use -pred_mode clas -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode selfattn>clas_train_hub.log &
CUDA_VISIBLE_DEVICES=7 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out clas_hub --hub_module_embedding use -pred_mode clas -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode avg>clas_eval_hub.log &


CUDA_VISIBLE_DEVICES=8 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out match_hub --hub_module_embedding use -pred_mode match -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode selfattn>match_train_hub.log &
CUDA_VISIBLE_DEVICES=9 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match_hub --hub_module_embedding use -pred_mode match -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode avg>match_eval_hub.log &

CUDA_VISIBLE_DEVICES=10 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out match_simple_hub --hub_module_embedding use -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode selfattn>match_simple_train_hub.log &
CUDA_VISIBLE_DEVICES=11 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match_simple_hub --hub_module_embedding use -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 16 -nhl 3 -nel 3 -ndl 3 -nh 8 -ag_mode avg>match_simple_eval_hub.log &

