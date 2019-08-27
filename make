nvcc -x cu -arch=sm_35 -I /usr/mpi/gcc/mvapich2-1.7/include -std=c++11 -c Simulator.cu -o Simulator.o
nvcc -x cu -arch=sm_35 -I /usr/mpi/gcc/mvapich2-1.7/include -std=c++11 --maxrregcount=48 -c GPUBeam/Collide.cu -o Collide.o
nvcc -x c++ -arch=sm_35 -I /usr/mpi/gcc/mvapich2-1.7/include-std=c++11 -c Beam.cpp -o Beam.o
nvcc -x c++ -arch=sm_35 -I /usr/mpi/gcc/mvapich2-1.7/include -std=c++11 -c InputProcessing.cpp -o InputProcessing.o
nvcc -x c++ -arch=sm_35 -I /usr/mpi/gcc/mvapich2-1.7/include -std=c++11 -c util/Util.cpp -o Util.o
nvcc -arch=sm_35 -L /usr/mpi/gcc/mvapich2-1.7/lib -lmpich -lmpl Beam.o InputProcessing.o Util.o Simulator.o Collide.o -o app
