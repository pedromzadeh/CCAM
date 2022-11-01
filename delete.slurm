#!/bin/bash
#SBATCH --account=bcamley1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=zadeh@jhu.edu
#SBATCH --time=10:00:00
#SBATCH -o %A_%a.out

rm output/ -rf