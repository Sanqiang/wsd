# Experiment for pred mode
#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/home/zhaos5/projs/wsd/wsd_code"

CUDA_VISIBLE_DEVICES=0,1 nohup python ../../model/train.py -env aws -ngpus 2 -mode base -out clas -pred_mode clas -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>clas_train.log &
CUDA_VISIBLE_DEVICES=15 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out clas -pred_mode clas -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>clas_eval.log &


CUDA_VISIBLE_DEVICES=2,3 nohup python ../../model/train.py -env aws -ngpus 2 -mode base -out match -pred_mode match -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_train.log &
CUDA_VISIBLE_DEVICES=14 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match -pred_mode match -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_eval.log &

CUDA_VISIBLE_DEVICES=4,5 nohup python ../../model/train.py -env aws -ngpus 2 -mode base -out match_simple -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_simple_train.log &
CUDA_VISIBLE_DEVICES=13 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match_simple -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_simple_eval.log &


CUDA_VISIBLE_DEVICES=6,7 nohup python ../../model/train.py -env aws -ngpus 2 -mode base -out clas_hub --hub_module_embedding use -pred_mode clas -max_context_len 1000 -dim 256 -bsize 8 -nhl 3 -nel 3 -nh 4 -ag_mode avg>clas_train_hub.log &
CUDA_VISIBLE_DEVICES=12 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out clas_hub --hub_module_embedding use -pred_mode clas -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>clas_eval_hub.log &


CUDA_VISIBLE_DEVICES=8 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out match_hub --hub_module_embedding use -pred_mode match -max_context_len 1000 -dim 256 -bsize 8 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_train_hub.log &
CUDA_VISIBLE_DEVICES=11 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match_hub --hub_module_embedding use -pred_mode match -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_eval_hub.log &

CUDA_VISIBLE_DEVICES=9 nohup python ../../model/train.py -env aws -ngpus 1 -mode base -out match_simple_hub --hub_module_embedding use -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 6 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_simple_train_hub.log &
CUDA_VISIBLE_DEVICES=10 nohup python ../../model/eval.py -env aws -ngpus 1 -mode base -out match_simple_hub --hub_module_embedding use -pred_mode match_simple -max_context_len 1000 -dim 256 -bsize 12 -nhl 3 -nel 3 -nh 4 -ag_mode avg>match_simple_eval_hub.log &

