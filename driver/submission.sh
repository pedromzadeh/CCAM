#!/bin/bash
echo "Submitting SVA jobs..."
sbatch --array=0-99 sva_job.slurm
sleep 5
sbatch --array=100-199 sva_job.slurm
sleep 5
sbatch --array=200-299 sva_job.slurm
sleep 5
sbatch --array=300-399 sva_job.slurm
sleep 5
sbatch --array=400-461 sva_job.slurm

echo "Submitting FFCR jobs..."
sbatch --array=0-99 ffcr_job.slurm
sleep 5
sbatch --array=100-199 ffcr_job.slurm
sleep 5
sbatch --array=200-299 ffcr_job.slurm
sleep 5
sbatch --array=300-399 ffcr_job.slurm
sleep 5
sbatch --array=400-461 ffcr_job.slurm
