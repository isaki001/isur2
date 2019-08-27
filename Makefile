CC=nvcc
hellomake: 
	$(CC) -x cu -arch=sm_35 -I /usr/mpi/gcc/openmpi-1.8.8/include/ -c Simulator.cu -o Simulator.o
	$(CC) -x cu  -arch=sm_35 -I /usr/mpi/gcc/openmpi-1.8.8/include/ --maxrregcount=48 -c GPUBeam/Collide.cu -o Collide.o
	$(CC) -x c++  -arch=sm_35 -I /usr/mpi/gcc/openmpi-1.8.8/include/ -c Beam.cpp -o Beam.o
	$(CC) -x c++  -arch=sm_35 -I /usr/mpi/gcc/openmpi-1.8.8/include/  -c InputProcessing.cpp -o InputProcessing.o
	$(CC) -x c++ -arch=sm_35 -I /usr/mpi/gcc/openmpi-1.8.8/include/ -c util/Util.cpp -o Util.o
	$(CC) -arch=sm_35 -L /usr/mpi/gcc/openmpi-1.8.8/lib -lmpi Beam.o InputProcessing.o Util.o Simulator.o Collide.o -o app