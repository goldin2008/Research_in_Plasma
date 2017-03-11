#!/bin/sh
#SBATCH --array=0-431
#SBATCH --time=05:20:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=4096       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=job
#SBATCH --mail-type=ALL         # notifications for job done & fail
#SBATCH --mail-user=lei.yu@huskers.unl.edu      # send-to address

# $i is for input
# 1 is for print output in program
# 2 is for err in program
for i in {0..431}
  do
    # nohup python position.py $i 1> out 2>err
    nohup python run_linear.py $i 1> out 2>err
    echo $i
done
