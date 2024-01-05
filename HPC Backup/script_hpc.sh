#!/bin/sh
# SBATCH -J "Run test"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=100G
#SBATCH -A hpc-prf-bbam
#SBATCH -p normal
#SBATCH -t 0-100:00:00
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

module reset
module lang/Anaconda3/2022.05

source activate /scratch/hpc-prf-bbam/avinashk/.conda/envs/master_thesis
cd /scratch/hpc-prf-bbam/avinashk/Brain-Models/examples/Won2022/

module list
python3 single_session_SOA_close_set.py
deactivate
exit 0
~