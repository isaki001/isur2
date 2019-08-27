# SET UP ENVIRONMENT
# This is CUDA
CUDA_INSTALL_PATH=/usr/local/cuda-7.0
MPIHOME=/usr/mpi/gcc/mvapich2-1.7

export PATH=${CUDA_INSTALL_PATH}/bin:${MPIHOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_INSTALL_PATH}/lib64:${CUDA_INSTALL_PATH}/lib:${MPIHOME}/lib:${MPIHOME}/lib64:/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

