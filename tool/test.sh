#!/bin/bash
#SBATCH -J gcn
#SBATCH -p ty_xd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1

# sbatch -p ty_xd -J gcn -N 1 -n 8 --gres=dcu:1 slurm.sh

#load env
module purge
module load compiler/devtoolset/7.3.1  mpi/hpcx/2.11.0/gcc-7.3.1  compiler/dtk/23.04
# module list
# module unload compiler/dtk/21.10
# module load compiler/dtk/22.04
# module list

# echo ${SLURM_JOB_NAME}
#CUDA code to HIP code

# hipify-perl gcn.cu > gcn.cpp
#to compile HIP code
hipcc test_bank_c_r_major.cpp -o test_bank_c_r_major

#运行测试代码
## for correctness

hipprof --hip-trace ./test_bank_c_r_major