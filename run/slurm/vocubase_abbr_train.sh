#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=wsd_vocubase_abbr_train
#SBATCH --output=wsd_vocubase_abbr_train.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/zfs1/hdaqing/saz31/wsd/wsd_code"

# Run the job
srun python ../../model/train.py -env crc -lr 0.01 -ngpus 1 -mode voc -out wsd_vocubase_abbr -pred_mode match -max_context_len 10 -dim 100 -bsize 256 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode abbr --model_print_freq 10 --train_emb false --voc_process add_abbr --encoder_mode abbr_ut2t --warm_start /zfs1/hdaqing/saz31/wsd/wsd_perf/ckpt/model.ckpt-30329
