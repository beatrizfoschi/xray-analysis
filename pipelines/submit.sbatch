#!/bin/bash
#SBATCH --job-name=nmf_xeol
##SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --constraint='hpc6|hpc7|hpc8'
#SBATCH --time=02:00:00
#SBATCH --output=nmf_%j.out
#SBATCH --error=nmf_%j.err




source ~/miniconda/etc/profile.d/conda.sh
conda activate lauexplore-env


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

##  --h5path /data/visitor/blc16817/bm32/20260120/RAW_DATA/MLed/MLed_map/MLed_map.h5 \
##  --scan-number 1 \
## seed 1960596962

python -u nmf_sbatch_job.py \
  --spectra-npy data_nmf.npy \
  --wl-npy wl_array.npy \
  --map-nx 81 --map-ny 121 \
  --n-components 5 \
  --loss kullback-leibler \
  --init random \
  --max-iter 5000 \
  --extent 0 40 40 0 \
  --random-state 0 \
  --outdir ./k_5/nmf_out_k_5_random \
  --tol 1e-5 \


##python nmf_xeol_job.py \  # <- para fazer load dos dados
## --spectra-npy spectra.npy \
##  --wl-npy wl_array.npy \
##  --map-nx 101 --map-ny 201 \
##  --n-components 3 \
##  --loss kullback-leibler \
##  --max-iter 2000 \
##  --outdir nmf_out
