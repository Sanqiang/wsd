#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=wsd_extra_train
#SBATCH --output=wsd_extra_train.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/zfs1/hdaqing/saz31/wsd/wsd_code"

# Run the job
srun python ../../model/train.py -env crc -lr 0.001 -ngpus 1 -mode base -out wsd_extra -pred_mode match -max_context_len 50 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode abbr --extra_loss def:stype --model_print_freq 2 --task_iter_steps 300 --cui_iter_steps 50
