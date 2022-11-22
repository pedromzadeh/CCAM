echo "Removing output/"
rm ../output -rf

echo "Finished removing output. Now running 48 collisions per submission"
sbatch --array=0-89 job.slurm
sqme