#!/bin/bash
#SBATCH --account=bcamley1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=end
#SBATCH --mail-user=zadeh@jhu.edu
#SBATCH --time=1:00:00
#SBATCH -o %A_%a.out

python process_data.py