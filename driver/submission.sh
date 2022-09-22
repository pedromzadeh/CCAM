#!/bin/bash
sbatch --array=0-99 job.slurm
sleep 5
sbatch --array=100-199 job.slurm
sleep 5
sbatch --array=200-299 job.slurm
sleep 5
sbatch --array=300-399 job.slurm
sleep 5
sbatch --array=400-461 job.slurm
