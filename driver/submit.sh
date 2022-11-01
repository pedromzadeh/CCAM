echo "Removing output/"
rm ../output -rf

echo "Finished removing output. Now running 11 * 48 collisions"
sbatch --array=0-10 prw_job.slurm
sqme