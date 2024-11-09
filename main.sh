#!/bin/bash
#SBATCH -J nextqa            #Job name(--job-name)
#SBATCH -e %j.err_          #Name of stdout output file(--output)
#SBATCH -o %j.out_          #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name ; available options "shared,medium,large or gpu" 
#SBATCH -N 1                       # no of Nodes
#SBATCH -n 4                       # no of processes or tasks
#SBATCH --gres=gpu:1               # request gpu card: it should be either 1 or 2
#SBATCH --cpus-per-task=4          # no of threads per process or task
#SBATCH -t 02:40:00         # specifies walltime(--time maximum duration)of run
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH --mail-user=tawseeq@kgpian.iitkgp.ac.in        # user's email ID where job status info will be sent

module load compiler/cuda/10.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate videoqa
python /scratch/22ch10090/visionlanguage/test1/NExT-QA/main_qa.py

