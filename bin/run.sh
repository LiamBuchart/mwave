#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lbuchart@eoas.ubc.ca
#SBATCH --account=rrg-rstull

ml StdEnv/2020  intel/2020.1.217  openmpi/4.0.3
module load wrf/4.2.1

cd ../exps/neutstab

#rm -r namelist.input
#ln -sv namelist.input.spinup namelist.input

srun ./wrf.exe 1>wrf.log 2>&1

mkdir -p log/spinup
mv rsl.* log/spinup/
mv wrf.log log/spinup

#rm -r namelist.input
#ln -sv namelist.input.restart namelist.input

#srun ./wrf.exe 1>wrf.log 2>&1

#mkdir -p log/restart
#mv rsl.* log/restart/
#mv wrf.log log/restart/
