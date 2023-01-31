# echo "Removing output/"
# rm ../output -rf

# echo "Finished removing output. Now running 48 collisions per submission"

# To seed each grid_id differently based on time, sleep for 1s when submitting
# grid_id jobs
# for i in {0..89}
# do
#     sbatch --array=$i job.slurm
#     sleep 1
# done

sbatch --array=$1-$2 job.slurm
sqme
