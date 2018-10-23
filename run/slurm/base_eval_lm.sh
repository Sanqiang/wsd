#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=wsd_base_eval_lm
#SBATCH --output=wsd_base_eval_lm.out
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
srun python ../../model/eval.py -env crc -ngpus 1 -mode base -out wsd_base_lm -pred_mode match -max_context_len 50 -dim 256 -bsize 128 -nhl 3 -nel 3 -nh 4 -ag_mode avg -it true --abbr_mode abbr
