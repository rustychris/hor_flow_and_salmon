#!/bin/bash -l
# NOTE the -l flag!

#SBATCH --job-name horcfg13
# Standard out and Standard Error output files with the job number in the name.
#SBATCH -o slurm_out-%j.output
#SBATCH -e slurm_out-%j.output
#SBATCH --partition high
#SBATCH --verbose

#SBATCH --mem-per-cpu 2G
#SBATCH --time 10-00:00:00
#SBATCH -n 16

conda activate general

python ./hor_snubby_oper.py -l -d runs/large13 -s 2018-03-10 -e 2018-04-15 -i 10D
# python ./hor_snubby_oper.py -l -d runs/largetest01 -s 2018-04-05T12:00 -e 2018-04-05T16:00

