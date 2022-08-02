#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=rrg-rstull
#SBATCH --time=01:00:00
#SBATCH --mem=4G  ## memory to ask for
#SBATCH --job-name="WRF-Python-2D"
#SBATCH --mail-user=lbuchart@eoas.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --output=sim-%j.out
#SBATCH --error=sim-%j.err
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run

module load python/3.8.10 scipy-stack mpi4py
source ~/projects/rrg-rstull/lbuchart/PYTHON/WRF-Analysis/bin/activate

python ~/projects/rrg-rstull/lbuchart/PYTHON/plot_wrf_sfc.py
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"

