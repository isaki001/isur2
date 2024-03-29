
#define BLOCKDIMX 128
#define WARP 32
#define NUM_WARPS (BLOCKDIMX/WARP)
#define ILP 4
#define PARTICLES_PER_THREAD ILP

#define META_TAG 1

#include <stdio.h>
//x,2
__device__
double
power(double x, int a){
  //return pow(x, a);  
  double val = 1;
  for(int i = 1; i <= a; ++i){
    val *= x;
  }
  return val;  
}


#if GCC_VERSION > 40800

void dumpBeamByThread(Simulator *sim, double *&d_x, double *&h_x, int Npart, int Ncol, int iTurn, std::string ic, std::_Ios_Openmode mode, int eORp){
  BeamParams *bParams = sim->bParams;
  InputProcessing *io = sim->io;

  int iTurn_after_offset =  iTurn - bParams->NdumpOffset;  
  if (iTurn == bParams->Niter || ( (iTurn_after_offset > 0) && (bParams->Nfreq > 0) && (iTurn_after_offset % bParams->Nfreq == 0 || io->pending_log))) {
    std::stringstream ss;
    ss << std::setprecision(16);
    ss << "Rank: " << sim->rank << " ==> " << iTurn << " turns finished...\n";
    std::cout << ss.str() << std::endl;
    if (io->log_in_progress) {
      if (bParams->strict_freq) {
	io->log_thread.join();
      } else {
	io->pending_log = true;
	return;
      }
    }
    if (io->log_thread.joinable()) { io->log_thread.join(); }
    io->pending_log = false;
    io->log_in_progress = true;

    if (eORp == 1) {
    QuadDebug(cudaMemcpy(h_x, d_x, sizeof(double) * bParams->pbunches * (NCOL + 2) * Npart, cudaMemcpyDeviceToHost));
    io->log_thread = std::thread(&InputProcessing::dumpParticles, io, bParams->pbunches, h_x, Npart, Ncol, bParams->Nfreq, iTurn, ic ,mode, sim->rank);
    }

    else {
    QuadDebug(cudaMemcpy(h_x, d_x, sizeof(double) * (bParams->ebunches) * (NCOL + 2) * Npart, cudaMemcpyDeviceToHost));
    io->log_thread = std::thread(&InputProcessing::dumpParticles, io, bParams->ebunches, h_x, Npart, Ncol, bParams->Nfreq, iTurn, ic ,mode, sim->rank);
    }

    if (!bParams->log_in_background && io->log_thread.joinable()) { io->log_thread.join(); }
  }   
  
}
#else

void dumpBeamByThread(Simulator *sim, double *&d_x, double *&h_x, int Npart, int Ncol, int iTurn, std::string ic, std::_Ios_Openmode mode, int eORp){
  BeamParams *bParams = sim->bParams;
  InputProcessing *io = sim->io;
  int iTurn_after_offset =  iTurn - bParams->NdumpOffset;  
  if (iTurn == bParams->Niter || ( (iTurn_after_offset > 0) && (bParams->Nfreq > 0) && (iTurn_after_offset % bParams->Nfreq == 0 || io->pending_log))) {
    std::stringstream ss;
    ss << std::setprecision(16);
    ss << "Rank: " << sim->rank << " ==> " << iTurn << " turns finished...\n";
    std::cout << ss.str() << std::endl;

    if (eORp == 1) {
    QuadDebug(cudaMemcpy(h_x, d_x, sizeof(double) * bParams->pbunches * (NCOL + 2) * Npart, cudaMemcpyDeviceToHost));
    io->dumpParticles(bParams->pbunches, h_x, Npart, Ncol, bParams->Nfreq, iTurn, ic , mode, sim->rank);
    }

    else {
    QuadDebug(cudaMemcpy(h_x, d_x, sizeof(double) * (bParams->ebunches) * (NCOL + 2) * Npart, cudaMemcpyDeviceToHost));
    io->dumpParticles(bParams->ebunches, h_x, Npart, Ncol, bParams->Nfreq, iTurn, ic , mode, sim->rank);
    }

  }
}

#endif


void
initDeviceMemory(int *&ditM, int *&hitM,
		 int *&dNrow, int *&hNrow,
		 double *&dx, double *&x,
                 int maxLen, int Npart, int Ncol){
  QuadDebug(cudaMalloc((void **)&ditM, sizeof(int) * maxLen * Ncol * (Ncol + 8)));
  QuadDebug(cudaMalloc((void **)&dNrow, sizeof(int) * Ncol));
  QuadDebug(cudaMalloc((void **)&dx, sizeof(double) * Npart * (Ncol + 2)));


  QuadDebug(cudaMemcpy(dNrow, hNrow, sizeof(int) * Ncol, cudaMemcpyHostToDevice));
  QuadDebug(cudaMemcpy(dx, x, sizeof(double) * (Ncol + 2) * Npart, cudaMemcpyHostToDevice));
  QuadDebug(cudaMalloc((void **)&ditM, sizeof(int) * maxLen * Ncol * (Ncol + 2)));
  QuadDebug(cudaMemcpy(ditM, hitM, sizeof(int) * maxLen * Ncol * (Ncol + 2), cudaMemcpyHostToDevice));  
}

void
initDeviceMemory_bunch(int *&ditM, int *&hitM,
                 int *&dNrow, int *&hNrow,
                 double *&dx, double *&x,
                 int maxLen, int Npart, int Ncol, int Nbunches, int eORp){
  QuadDebug(cudaMalloc((void **)&ditM, sizeof(int) * maxLen * Ncol * (Ncol + 8)));
  QuadDebug(cudaMalloc((void **)&dNrow, sizeof(int) * Ncol));

  /*if (eORp == 1)
  QuadDebug(cudaMalloc((void **)&dx, sizeof(double) * Nbunches * Npart * (Ncol + 2)));
  else
  QuadDebug(cudaMalloc((void **)&dx, sizeof(double) * (Nbunches - 1) * Npart * (Ncol + 2)));

  QuadDebug(cudaMemcpy(dNrow, hNrow, sizeof(int) * Ncol, cudaMemcpyHostToDevice));

  if (eORp == 1)
  QuadDebug(cudaMemcpy(dx, x, sizeof(double) * Nbunches * (Ncol + 2) * Npart, cudaMemcpyHostToDevice));
  else
  QuadDebug(cudaMemcpy(dx, x, sizeof(double) * (Nbunches - 1) * (Ncol + 2) * Npart, cudaMemcpyHostToDevice)); */

  QuadDebug(cudaMalloc((void **)&ditM, sizeof(int) * maxLen * Ncol * (Ncol + 2)));
  QuadDebug(cudaMemcpy(ditM, hitM, sizeof(int) * maxLen * Ncol * (Ncol + 2), cudaMemcpyHostToDevice));
}


//@brief Template function to display GPU device array variables
template <class K>
void display(K *array, size_t size){
  K *tmp = (K *)malloc(sizeof(K) * size);
  cudaMemcpy(tmp, array, sizeof(K)*size, cudaMemcpyDeviceToHost);
  for(int i = 0 ; i < size; ++i){
    printf("%.16e \n",(double)tmp[i]);
  }
}

//Apply map for each attribute
__device__
double
applyMapByAtrib(int const* __restrict__ itM, double x0, double x1, double x2, double x3, double x4, double x5,
		int Npart, int iNrow, int maxLen, int attribId){
  extern __shared__ int sdata[];

  //int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  int laneId = threadIdx.x%WARP;
  int warpId = threadIdx.x/WARP;
  
  int cDataIdx = attribId * maxLen * (NCOL + 2);
  double mxterm = 0;

  int numLocalBlocks = (iNrow * (NCOL + 2))/(BLOCKDIMX);
  
  for(int i = 0; i < numLocalBlocks; ++i){
    int startIndex = cDataIdx + i * BLOCKDIMX;    
    sdata[threadIdx.x] = itM[startIndex + threadIdx.x];
    
    //Number of warps
    int numWarps = BLOCKDIMX/WARP;
    for(int j = 0; j < numWarps; ++j){
      int newWarpId = (warpId + j)%numWarps;
      //int newWarpId = (warpId + j);
      //newWarpId = newWarpId &(numWarps - 1);
      int it = sdata[newWarpId * WARP + laneId];
      
      // ie < Npart may not work correctly, since you are using shfl you need 32 threads
      for(int k = 0; (k < 4); ++k){ 
	double xterm = 1;
	int idx = (NCOL + 2) * k;

	
	int i0 = __shfl(it, idx + 0);
	int i1 = __shfl(it, idx + 1);
	int i2 = __shfl(it, idx + 2);
	int i3 = __shfl(it, idx + 3);
	int i4 = __shfl(it, idx + 4);
	int i5 = __shfl(it, idx + 5);
	int lo = __shfl(it, idx + 6);
	int hi = __shfl(it, idx + 7);

	// Recreate the 64b number.
	double m = 0;
	asm volatile("mov.b64 %0, {%1, %2};" : "=d"(m) : "r"(lo), "r"(hi));
	
	xterm *= power(x0, i0);
	
	xterm *= power(x1, i1);
	
	xterm *= power(x2, i2);
	
	xterm *= power(x3, i3);
	
	xterm *= power(x4, i4);
	
	xterm *= power(x5, i5);
	

	mxterm = mxterm + xterm * m;
      }
      __syncthreads();      
    }
    __syncthreads();
  }

  
  //Last block in itM
  {
    int remainingData = iNrow * (NCOL + 2) - numLocalBlocks * BLOCKDIMX;
    if(threadIdx.x < remainingData){
      int startIndex = cDataIdx + numLocalBlocks * BLOCKDIMX;
      sdata[threadIdx.x] = itM[startIndex + threadIdx.x];
    }
    __syncthreads();
    
    int remFullWarps = remainingData/WARP;
    
    for(int j = 0; j < remFullWarps; j++){ 
      int it = sdata[j * WARP + laneId];
      for(int k = 0; (k < 4); ++k){
	double xterm = 1;
	int idx = (NCOL + 2) * k;

	int i0 = __shfl(it, idx + 0);
	xterm *= power(x0, i0);
	int i1 = __shfl(it, idx + 1);
	xterm *= power(x1, i1);
	int i2 = __shfl(it, idx + 2);
	xterm *= power(x2, i2);
	int i3 = __shfl(it, idx + 3);
	xterm *= power(x3, i3);
	int i4 = __shfl(it, idx + 4);
	xterm *= power(x4, i4);
	int i5 = __shfl(it, idx + 5);
	xterm *= power(x5, i5);
	
	int lo = __shfl(it, idx + 6);
	int hi = __shfl(it, idx + 7);
	
	// Recreate the 64b number.
	double m;
	asm volatile("mov.b64 %0, {%1, %2};" : "=d"(m) : "r"(lo), "r"(hi));
	mxterm = mxterm + xterm * m;
      }
    }
    
    //Last partial WARP
    remainingData = remainingData - remFullWarps * WARP;
    int it = 0;
    if(laneId < remainingData){
      it = sdata[remFullWarps * WARP + laneId];
    }
    __syncthreads();      
    for(int k = 0; (k < remainingData/(NCOL + 2)); ++k){
      double xterm = 1;
      int idx = (NCOL + 2) * k;
      int i0 = __shfl(it, idx + 0);
      xterm *= power(x0, i0);
      int i1 = __shfl(it, idx + 1);
      xterm *= power(x1, i1);
      int i2 = __shfl(it, idx + 2);
      xterm *= power(x2, i2);
      int i3 = __shfl(it, idx + 3);
      xterm *= power(x3, i3);
      int i4 = __shfl(it, idx + 4);
      xterm *= power(x4, i4);
      int i5 = __shfl(it, idx + 5);
      xterm *= power(x5, i5);
      
      int lo = __shfl(it, idx + 6);
      int hi = __shfl(it, idx + 7);
	
      // Recreate the 64b number.
      double m;
      asm volatile("mov.b64 %0, {%1, %2};" : "=d"(m) : "r"(lo), "r"(hi));
      mxterm = mxterm + xterm * m;
    }
  }
  __syncthreads();
  
  return mxterm;
}




__device__
void
applyMapByAtrib(int const* __restrict__ itM, 
		/*double *x, double *op,*/
		double &x0, double x1, double x2, double x3, double x4, double x5,
		double &y0, double y1, double y2, double y3, double y4, double y5,
		double &z0, double z1, double z2, double z3, double z4, double z5,
		double &w0, double w1, double w2, double w3, double w4, double w5,
		int Npart, int iNrow, int maxLen, int attribId){
  extern __shared__ int sdata[];
  
  int laneId = threadIdx.x%WARP;
  int warpId = threadIdx.x/WARP;

  int cDataIdx = attribId * maxLen * (NCOL + 2);
  double mxterm = 0, myterm = 0, mzterm = 0, mwterm = 0;

  int numLocalBlocks = (iNrow * (NCOL + 2))/(BLOCKDIMX);

  for(int i = 0; i < numLocalBlocks; ++i){
    int startIndex = cDataIdx + i * BLOCKDIMX;
    sdata[threadIdx.x] = itM[startIndex + threadIdx.x];

    //Number of warps
    int numWarps = BLOCKDIMX/WARP;
    
    //#pragma unroll 4
    for(int j = 0; j < numWarps; ++j){
      int newWarpId = (warpId + j)%numWarps;
      //int newWarpId = (warpId + j);
      //newWarpId = newWarpId &(numWarps - 1);
      int it = sdata[newWarpId * WARP + laneId];

      // ie < Npart may not work correctly, since you are using shfl you need 32 threads
#pragma unroll 4
      for(int k = 0; (k < 4); ++k){
        double xterm = 1, yterm = 1, zterm = 1, wterm = 1;
        int idx = (NCOL + 2) * k;

	int i0 = __shfl(it, idx + 0);
        int i1 = __shfl(it, idx + 1);
	for(int ii = 1; ii <= i0; ++ii){
	  xterm *= x0;
	  yterm *= y0;
	  zterm *= z0;
	  wterm *= w0;
	}
	for(int ii = 1; ii <= i1; ++ii){
	  xterm *= x1;
	  yterm *= y1;
	  zterm *= z1;
	  wterm *= w1;
	}
	
	int i2 = __shfl(it, idx + 2);
        int i3 = __shfl(it, idx + 3);
        for(int ii = 1; ii <= i2; ++ii){
	  xterm *= x2;
	  yterm *= y2;
	  zterm *= z2;
	  wterm *= w2;
	}
	for(int ii = 1; ii <= i3; ++ii){
	  xterm *= x3;
	  yterm *= y3;
	  zterm *= z3;
	  wterm *= w3;
	}
	
	int i4 = __shfl(it, idx + 4);
        int i5 = __shfl(it, idx + 5);
	for(int ii = 1; ii <= i4; ++ii){
	  xterm *= x4;
	  yterm *= y4;
	  zterm *= z4;
	  wterm *= w4;
	}
	for(int ii = 1; ii <= i5; ++ii){
	  xterm *= x5;
	  yterm *= y5;
	  zterm *= z5;
	  wterm *= w5;
	}

	int lo = __shfl(it, idx + 6);
        int hi = __shfl(it, idx + 7);

        // Recreate the 64b number.
        double m = 0;
        asm volatile("mov.b64 %0, {%1, %2};" : "=d"(m) : "r"(lo), "r"(hi));
	mxterm = mxterm + xterm * m;
	myterm = myterm + yterm * m;	
	mzterm = mzterm + zterm * m;	
	mwterm = mwterm + wterm * m;	
      }
      __syncthreads();
    }
    __syncthreads();
  }

  
  //Last block in itM
  {
    int remainingData = iNrow * (NCOL + 2) - numLocalBlocks * BLOCKDIMX;
    if(threadIdx.x < remainingData){
      int startIndex = cDataIdx + numLocalBlocks * BLOCKDIMX;
      sdata[threadIdx.x] = itM[startIndex + threadIdx.x];
    }
    __syncthreads();

    int remFullWarps = remainingData/WARP;

    for(int j = 0; j < remFullWarps; j++){
      int it = sdata[j * WARP + laneId];
      for(int k = 0; (k < 4); ++k){
        double xterm = 1, yterm = 1, zterm = 1, wterm = 1;
        int idx = (NCOL + 2) * k;

        int i0 = __shfl(it, idx + 0);
        int i1 = __shfl(it, idx + 1);       
	for(int ii = 1; ii <= i0; ++ii){
	  xterm *= x0;
	  yterm *= y0;
	  zterm *= z0;
	  wterm *= w0;
	}
	for(int ii = 1; ii <= i1; ++ii){
	  xterm *= x1;
	  yterm *= y1;
	  zterm *= z1;
	  wterm *= w1;
	}
	
	int i2 = __shfl(it, idx + 2);
        int i3 = __shfl(it, idx + 3);
        for(int ii = 1; ii <= i2; ++ii){
	  xterm *= x2;
	  yterm *= y2;
	  zterm *= z2;
	  wterm *= w2;
	}
	for(int ii = 1; ii <= i3; ++ii){
	  xterm *= x3;
	  yterm *= y3;
	  zterm *= z3;
	  wterm *= w3;
	}
	
	int i4 = __shfl(it, idx + 4);
        int i5 = __shfl(it, idx + 5);
	for(int ii = 1; ii <= i4; ++ii){
	  xterm *= x4;
	  yterm *= y4;
	  zterm *= z4;
	  wterm *= w4;
	}
	for(int ii = 1; ii <= i5; ++ii){
	  xterm *= x5;
	  yterm *= y5;
	  zterm *= z5;
	  wterm *= w5;
	}

        int lo = __shfl(it, idx + 6);
        int hi = __shfl(it, idx + 7);

        // Recreate the 64b number.
        double m;
        asm volatile("mov.b64 %0, {%1, %2};" : "=d"(m) : "r"(lo), "r"(hi));
        mxterm = mxterm + xterm * m;
	myterm = myterm + yterm * m;	
	mzterm = mzterm + zterm * m;	
	mwterm = mwterm + wterm * m;	
      }
    }

    //Last partial WARP
    remainingData = remainingData - remFullWarps * WARP;
    int it = 0;
    if(laneId < remainingData){
      it = sdata[remFullWarps * WARP + laneId];
    }
    __syncthreads();

    for(int k = 0; (k < remainingData/(NCOL + 2)); ++k){
      double xterm = 1, yterm = 1, zterm = 1, wterm = 1;
      int idx = (NCOL + 2) * k;

      int i0 = __shfl(it, idx + 0);
      int i1 = __shfl(it, idx + 1);
      for(int ii = 1; ii <= i0; ++ii){
	xterm *= x0;
	yterm *= y0;
	zterm *= z0;
	wterm *= w0;
      }
      for(int ii = 1; ii <= i1; ++ii){
	xterm *= x1;
	yterm *= y1;
	zterm *= z1;
	wterm *= w1;
      }
      
      int i2 = __shfl(it, idx + 2);
      int i3 = __shfl(it, idx + 3);
      for(int ii = 1; ii <= i2; ++ii){
	xterm *= x2;
	yterm *= y2;
	zterm *= z2;
	wterm *= w2;
      }
      for(int ii = 1; ii <= i3; ++ii){
	xterm *= x3;
	yterm *= y3;
	zterm *= z3;
	wterm *= w3;
      }
      
      int i4 = __shfl(it, idx + 4);
      int i5 = __shfl(it, idx + 5);
      for(int ii = 1; ii <= i4; ++ii){
	xterm *= x4;
	yterm *= y4;
	zterm *= z4;
	wterm *= w4;
      }
      for(int ii = 1; ii <= i5; ++ii){
	xterm *= x5;
	yterm *= y5;
	zterm *= z5;
	wterm *= w5;
      }


      int lo = __shfl(it, idx + 6);
      int hi = __shfl(it, idx + 7);

      // Recreate the 64b number.
      double m;
      asm volatile("mov.b64 %0, {%1, %2};" : "=d"(m) : "r"(lo), "r"(hi));
      mxterm = mxterm + xterm * m; 
      myterm = myterm + yterm * m;	
      mzterm = mzterm + zterm * m;	
      mwterm = mwterm + wterm * m;	
    }
  }
  
  __syncthreads();

  x0 = mxterm;
  y0 = myterm;
  z0 = mzterm;
  w0 = mwterm;
}

#define OUT_OF_BOUND(x, y, maxX, maxY) ((abs(x) > maxX) || (abs(y) > maxY))
__global__
void
applyMSHFL(int *outOfBound, int const* __restrict__ itM, double *x, double *op, int Npart, int maxLen, int iNrow, int attribId, double x_bound, double y_bound){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
  double y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0, y5 = 0;
  double z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0;
  double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0;
  
  if(ie < Npart/PARTICLES_PER_THREAD){
    x0 = x[ie];
    x1 = x[Npart + ie];
    x2 = x[Npart * 2 + ie];
    x3 = x[Npart * 3 + ie];
    x4 = x[Npart * 4 + ie];
    x5 = x[Npart * 5 + ie];

    if(OUT_OF_BOUND(x0, x2, x_bound, y_bound) || outOfBound[ie] == 1){
      x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
      outOfBound[ie] = 1;
    }
#if ILP > 1    
    ie += Npart/PARTICLES_PER_THREAD;
    y0 = x[ie];
    y1 = x[Npart + ie];
    y2 = x[Npart * 2 + ie];
    y3 = x[Npart * 3 + ie];
    y4 = x[Npart * 4 + ie];
    y5 = x[Npart * 5 + ie];    
    if(OUT_OF_BOUND(y0, y2, x_bound, y_bound) || outOfBound[ie] == 1){
      y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0, y5 = 0;
      outOfBound[ie] = 1;
    }
#endif
#if ILP > 2    
    ie += Npart/PARTICLES_PER_THREAD;
    z0 = x[ie];
    z1 = x[Npart + ie];
    z2 = x[Npart * 2 + ie];
    z3 = x[Npart * 3 + ie];
    z4 = x[Npart * 4 + ie];
    z5 = x[Npart * 5 + ie];    
    if(OUT_OF_BOUND(z0, z2, x_bound, y_bound) || outOfBound[ie] == 1){
      z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0;
      outOfBound[ie] = 1;
    }
#endif
#if ILP > 3
    ie += Npart/PARTICLES_PER_THREAD;
    w0 = x[ie];
    w1 = x[Npart + ie];
    w2 = x[Npart * 2 + ie];
    w3 = x[Npart * 3 + ie];
    w4 = x[Npart * 4 + ie];
    w5 = x[Npart * 5 + ie];
    if(OUT_OF_BOUND(w0, w2, x_bound, y_bound) || outOfBound[ie] == 1){
      w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0;
      outOfBound[ie] = 1;
    }
#endif
  }
  
  applyMapByAtrib(itM, /*x, op,*/ 
		  x0, x1, x2, x3, x4, x5,
		  y0, y1, y2, y3, y4, y5,
		  z0, z1, z2, z3, z4, z5,
		  w0, w1, w2, w3, w4, w5,
		  Npart, iNrow, maxLen, attribId);

  ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  if(ie < Npart/PARTICLES_PER_THREAD){
    op[attribId * Npart + ie] = x0;
#if ILP > 1
    op[attribId * Npart + Npart/PARTICLES_PER_THREAD + ie] = y0;
#endif
#if ILP > 2
    op[attribId * Npart + 2*Npart/PARTICLES_PER_THREAD + ie] = z0;
#endif
#if ILP > 3
    op[attribId * Npart + 3*Npart/PARTICLES_PER_THREAD + ie] = w0;
#endif
  }
  
}


__global__
void
standardized_moments(double *x, double *y, double *z, 
		     double xbar, double ybar, double zbar,
		     double sig_x, double sig_y, double sig_z,
		     double *mx1, double *my1, double *mz1,
		     double *mx2, double *my2, double *mz2,
		     int size){
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if(pid < size){
    double xs = 0, ys = 0, zs = 0;
    xs = x[pid];
    ys = y[pid];
    zs = z[pid];
	
    double m = (xs - xbar)/sig_x;
	//if(pid<100)
		//printf("mx[%i]:%f\n", pid, m);
    mx1[pid] = m * m * m;
    mx2[pid] = m * m * m * m;
    
    m = (ys - ybar)/sig_y;
	//if(pid<100)
	//	printf("my[%i]:%f\n", pid, m);
    my1[pid] = m * m * m;
    my2[pid] = m * m * m * m;

    m = (zs - zbar)/sig_z;
	//if(pid==500)
		//printf("mz[%i]:%f\n", pid, m);
    mz1[pid] = m * m * m;
    mz2[pid] = m * m * m * m;
  } 
}

__global__
void
sd_sq(double *x, double *y, double *z, 
      double *px, double *py, double *pz, 
      double xbar, double ybar, double zbar, 
      double pxbar, double pybar, double pzbar, 
      double *ox, double *oy, double *oz, 
      double *opx, double *opy, double *opz,
      int size){
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(pid < size){
    double xs = 0, ys = 0, zs = 0;
    double pxs = 0, pys = 0, pzs = 0;

    xs = x[pid];pxs = px[pid];
    ys = y[pid];pys = py[pid];
    zs = z[pid];pzs = pz[pid];

	//if(pid<100)
	{
		//printf("xs[%i]:%f xbar:%f\n", pid, xs, xbar);
		//printf("xs-xbar%f\n", xs-xbar);
		//printf("multiplication:%f\n", (xs - xbar) * (xs - xbar));
	}
    ox[pid] = (xs - xbar) * (xs - xbar);
	//if(pid==500)
		//printf("ox[500]:%f\n", ox[pid]);
    oy[pid] = (ys - ybar) * (ys - ybar);
    oz[pid] = (zs - zbar) * (zs - zbar);
    opx[pid] = (pxs - pxbar) * (pxs - pxbar);
    opy[pid] = (pys - pybar) * (pys - pybar);
    opz[pid] = (pzs - pzbar) * (pzs - pzbar);
  }
  
}



//!============================================================================
//! Compute the rms properties of the entire beam. 
//!----------------------------------------------------------------------------

// x - NCOL * Npart

//   @param dx - ICs in device array NCOL * Npart
 
void 
computeRMS(double *dx, int Npart, 
	   double &xbar, double &ybar, double &zbar,
	   double &sig_x, double &sig_y, double &sig_z,
	   double *mom_x, double *mom_y, double *mom_z,
	   double &pxbar, double &pybar, double &pzbar,
	   double &sig_px, double &sig_py, double &sig_pz, int Npart_inbound){

  thrust::device_ptr<double> ptr;
  ptr = thrust::device_pointer_cast(dx);
  xbar = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;
  ybar = thrust::reduce(ptr + 2 * Npart, ptr + 3 * Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;
  zbar = thrust::reduce(ptr + 4 * Npart, ptr + 5 * Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;

  pxbar = thrust::reduce(ptr + Npart, ptr + 2 * Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;
  pybar = thrust::reduce(ptr + 3 * Npart, ptr + 4 * Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;
  pzbar = thrust::reduce(ptr + 5 * Npart, ptr + 6 * Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;

  double *dx0 = 0, *dy0 = 0, *dz0 = 0;
  double *dx1 = 0, *dy1 = 0, *dz1 = 0;
  QuadDebug(cudaMalloc((void **)&dx0, sizeof(double) * Npart));
  QuadDebug(cudaMalloc((void **)&dy0, sizeof(double) * Npart));
  QuadDebug(cudaMalloc((void **)&dz0, sizeof(double) * Npart));
  QuadDebug(cudaMalloc((void **)&dx1, sizeof(double) * Npart));
  QuadDebug(cudaMalloc((void **)&dy1, sizeof(double) * Npart));
  QuadDebug(cudaMalloc((void **)&dz1, sizeof(double) * Npart));

  int numThreads = BLOCKDIMX;
  int numBlocks = Npart/numThreads + ((Npart%numThreads)?1:0);
  
  sd_sq<<<numBlocks, numThreads>>>(dx, &dx[2 * Npart], &dx[4 * Npart], 
				   &dx[Npart], &dx[3 * Npart], &dx[5 * Npart], 
				   xbar, ybar, zbar, 
				   pxbar, pybar, pzbar, 
				   dx0, dy0, dz0, 
				   dx1, dy1, dz1,
				   Npart);
  ptr = thrust::device_pointer_cast(dx0);
  sig_x = sqrt(thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound);
  
  ptr = thrust::device_pointer_cast(dy0);
  sig_y = sqrt(thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound);
  ptr = thrust::device_pointer_cast(dz0);
  sig_z = sqrt(thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound);
  ptr = thrust::device_pointer_cast(dx1);
  sig_px = sqrt(thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound);
  ptr = thrust::device_pointer_cast(dy1);
  sig_py = sqrt(thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound);
  ptr = thrust::device_pointer_cast(dz1);
  sig_pz = sqrt(thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound);

  standardized_moments<<<numBlocks, numThreads>>>(dx, &dx[2 * Npart], &dx[4 * Npart],
						  xbar, ybar, zbar,
						  sig_x, sig_y, sig_z,
						  dx0, dy0, dz0, 
						  dx1, dy1, dz1,
						  Npart);
  ptr = thrust::device_pointer_cast(dx0);
  mom_x[0] = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;
  ptr = thrust::device_pointer_cast(dy0);
  mom_y[0] = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;
  ptr = thrust::device_pointer_cast(dz0);
  mom_z[0] = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound;
  ptr = thrust::device_pointer_cast(dx1);
  mom_x[1] = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound - 3.0;
  ptr = thrust::device_pointer_cast(dy1);
  mom_y[1] = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound - 3.0;
  ptr = thrust::device_pointer_cast(dz1);
  mom_z[1] = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>())/(double)Npart_inbound - 3.0;

  QuadDebug(cudaFree(dx0));
  QuadDebug(cudaFree(dy0));
  QuadDebug(cudaFree(dz0));
  QuadDebug(cudaFree(dx1));
  QuadDebug(cudaFree(dy1));
  QuadDebug(cudaFree(dz1));
}



__device__
void
gf2EqnsJac(int const* __restrict__ itM, double *xi, int *dflag, 
	   double xf0, double xf1, double xf2, double xf3, double xf4, double xf5,
	   double *f_xi_xf, double *jac, double delta, int Npart, int maxLen, 
	   int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
  
  if(ie < Npart && dflag[ie] == 0){
    x0 = xi[ie];
    x1 = xf1;
    x2 = xi[Npart * 2 + ie];
    x3 = xf3;
    x4 = xi[Npart * 4 + ie];
    x5 = xf5;
  }


  double f0 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow0, maxLen, 0);
  double f1 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow1, maxLen, 1);
  double f2 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow2, maxLen, 2);
  double f3 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow3, maxLen, 3);
  double f4 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow4, maxLen, 4);
  double f5 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow5, maxLen, 5);


  if(ie < Npart  && dflag[ie] == 0){

    f0 = xf0 - f0;
    f1 = xi[1*Npart + ie] - f1;
    f2 = xf2 - f2;
    f3 = xi[3*Npart + ie] - f3;
    f4 = xf4 - f4;
    f5 = xi[5*Npart + ie] - f5;


    jac[ie] = (f0 - f_xi_xf[ie])/delta;
    jac[1*Npart + ie] = (f1 - f_xi_xf[1*Npart + ie])/delta;
    jac[2*Npart + ie] = (f2 - f_xi_xf[2*Npart + ie])/delta;
    jac[3*Npart + ie] = (f3 - f_xi_xf[3*Npart + ie])/delta;
    jac[4*Npart + ie] = (f4 - f_xi_xf[4*Npart + ie])/delta;
    jac[5*Npart + ie] = (f5 - f_xi_xf[5*Npart + ie])/delta;


  }
}

__global__
void
gf2Eqns(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, int Npart, int maxLen, int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
  
  if(ie < Npart){
    x0 = xi[ie];
    x1 = xf[Npart + ie];
    x2 = xi[Npart * 2 + ie];
    x3 = xf[Npart * 3 + ie];
    x4 = xi[Npart * 4 + ie];
    x5 = xf[Npart * 5 + ie];
  }

  double f0 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow0, maxLen, 0);
  double f1 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow1, maxLen, 1);
  double f2 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow2, maxLen, 2);
  double f3 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow3, maxLen, 3);
  double f4 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow4, maxLen, 4);
  double f5 = applyMapByAtrib(itM, x0, x1, x2, x3, x4, x5, Npart, iNrow5, maxLen, 5);

  
  if(ie < Npart){
    f0 = xf[ie] - f0;
    f_xi_xf[ie] = f0;

    f1 = xi[1*Npart + ie] - f1;
    f_xi_xf[1*Npart + ie] = f1;

   
    f2 = xf[2*Npart + ie] - f2;
    f_xi_xf[2*Npart + ie] = f2;

    f3 = xi[3*Npart + ie] - f3;
    f_xi_xf[3*Npart + ie] = f3;

    f4 = xf[4*Npart + ie] - f4;
    f_xi_xf[4*Npart + ie] = f4;
    
    f5 = xi[5*Npart + ie] - f5;
    f_xi_xf[5*Npart + ie] = f5;


  }
}

__global__
void
JacobianKernel0(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
		int Npart, int maxLen,
		int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double xf0 = 0, xf1 = 0, xf2 = 0, xf3 = 0, xf4 = 0, xf5 = 0, delta;
  
  if(ie < Npart && dflag[ie] == 0){
    xf0 = xf[ie];
    xf1 = xf[1*Npart + ie];
    xf2 = xf[2*Npart + ie];
    xf3 = xf[3*Npart + ie];
    xf4 = xf[4*Npart + ie];
    xf5 = xf[5*Npart + ie];

    delta = sqrt(PRECISION*fabs(xf0));
    if (delta==0) delta = PRECISION;    
    xf0 += delta;
  }
  
  gf2EqnsJac(itM, xi, dflag,
	     xf0, xf1, xf2, xf3, xf4, xf5,
	     f_xi_xf, &jac[0*6*Npart], delta, Npart, maxLen, 
	     iNrow0, iNrow1, iNrow2, iNrow3, iNrow4, iNrow5);

}

__global__
void
JacobianKernel1(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
		int Npart, int maxLen,
		int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double xf0 = 0, xf1 = 0, xf2 = 0, xf3 = 0, xf4 = 0, xf5 = 0, delta;
  
  if(ie < Npart && dflag[ie] == 0){
    xf0 = xf[ie];
    xf1 = xf[1*Npart + ie];
    xf2 = xf[2*Npart + ie];
    xf3 = xf[3*Npart + ie];
    xf4 = xf[4*Npart + ie];
    xf5 = xf[5*Npart + ie];

    delta = sqrt(PRECISION*fabs(xf1));
    if (delta==0) delta = PRECISION;    
    xf1 += delta;
  }
  

  gf2EqnsJac(itM, xi, dflag, 
	     xf0, xf1, xf2, xf3, xf4, xf5,
	     f_xi_xf, &jac[1*6*Npart], delta, Npart, maxLen, 
	     iNrow0, iNrow1, iNrow2, iNrow3, iNrow4, iNrow5);
}
  
__global__
void
JacobianKernel2(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
		int Npart, int maxLen,
		int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double xf0 = 0, xf1 = 0, xf2 = 0, xf3 = 0, xf4 = 0, xf5 = 0, delta;
  
  if(ie < Npart && dflag[ie] == 0){
    xf0 = xf[ie];
    xf1 = xf[1*Npart + ie];
    xf2 = xf[2*Npart + ie];
    xf3 = xf[3*Npart + ie];
    xf4 = xf[4*Npart + ie];
    xf5 = xf[5*Npart + ie];
    
    delta = sqrt(PRECISION*fabs(xf2));
    if (delta==0) delta = PRECISION;    
    xf2 += delta;
  }
  
  gf2EqnsJac(itM, xi,  dflag,
	     xf0, xf1, xf2, xf3, xf4, xf5,
	     f_xi_xf, &jac[2*6*Npart], delta, Npart, maxLen, 
	     iNrow0, iNrow1, iNrow2, iNrow3, iNrow4, iNrow5);
}

__global__
void
JacobianKernel3(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
		int Npart, int maxLen,
		int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double xf0 = 0, xf1 = 0, xf2 = 0, xf3 = 0, xf4 = 0, xf5 = 0, delta;
  
  if(ie < Npart && dflag[ie] == 0){
    xf0 = xf[ie];
    xf1 = xf[1*Npart + ie];
    xf2 = xf[2*Npart + ie];
    xf3 = xf[3*Npart + ie];
    xf4 = xf[4*Npart + ie];
    xf5 = xf[5*Npart + ie];

    delta = sqrt(PRECISION*fabs(xf3));
    if (delta==0) delta = PRECISION;    
    xf3 += delta;
  }
  
  gf2EqnsJac(itM, xi, dflag, 
	     xf0, xf1, xf2, xf3, xf4, xf5,
	     f_xi_xf, &jac[3*6*Npart], delta, Npart, maxLen, 
	     iNrow0, iNrow1, iNrow2, iNrow3, iNrow4, iNrow5);
}

__global__
void
JacobianKernel4(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
		int Npart, int maxLen,
		int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double xf0 = 0, xf1 = 0, xf2 = 0, xf3 = 0, xf4 = 0, xf5 = 0, delta;
  
  if(ie < Npart && dflag[ie] == 0){
    xf0 = xf[ie];
    xf1 = xf[1*Npart + ie];
    xf2 = xf[2*Npart + ie];
    xf3 = xf[3*Npart + ie];
    xf4 = xf[4*Npart + ie];
    xf5 = xf[5*Npart + ie];

    delta = sqrt(PRECISION*fabs(xf4));
    if (delta==0) delta = PRECISION;    
    xf4 += delta;
  }
  
  gf2EqnsJac(itM, xi, dflag, 
	     xf0, xf1, xf2, xf3, xf4, xf5,
	     f_xi_xf, &jac[4*6*Npart], delta, Npart, maxLen, 
	     iNrow0, iNrow1, iNrow2, iNrow3, iNrow4, iNrow5);
}
 
__global__
void
JacobianKernel5(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
		int Npart, int maxLen,
		int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double xf0 = 0, xf1 = 0, xf2 = 0, xf3 = 0, xf4 = 0, xf5 = 0, delta;
  
  if(ie < Npart && dflag[ie] == 0){
    xf0 = xf[ie];
    xf1 = xf[1*Npart + ie];
    xf2 = xf[2*Npart + ie];
    xf3 = xf[3*Npart + ie];
    xf4 = xf[4*Npart + ie];
    xf5 = xf[5*Npart + ie];

    delta = sqrt(PRECISION*fabs(xf5));
    if (delta==0) delta = PRECISION;    
    xf5 += delta;
  }
  
  gf2EqnsJac(itM, xi, dflag, 
	     xf0, xf1, xf2, xf3, xf4, xf5,
	     f_xi_xf, &jac[5*6*Npart], delta, Npart, maxLen, 
	     iNrow0, iNrow1, iNrow2, iNrow3, iNrow4, iNrow5);
  
}

__global__
void
scale(double *da, double *vv, int Npart){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  if(ie < Npart){
    double temp, big;
    for (int i = 0; i < 6; ++i){
      big = 0.0;
      for (int j = 0; j < 6; ++j){
	double a = da[Npart * (j * 6 + i) + ie];
	if((temp=fabs(a)) > big) big = temp;
      }
      vv[i*Npart + ie] = 1/big;
    }
  }
}


//TODO - Singular matrix check
__global__
void
LUDcmp_lubksb(double *da, double *f, double *xf, int *dflag, int Npart){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  int imax;
  double dum, sum;
  
  double vv[6];// = new double[6];
  double a[36];// = new double[36];
  int idx[6];

  if(ie < Npart && dflag[ie] == 0){
    double big, temp;
    for (int i = 0; i < 6; ++i){
      big = 0.0;
      for (int j = 0; j < 6; ++j){
	a[i * 6 + j] = da[Npart * (j * 6 + i) + ie];
	if((temp=fabs(a[i * 6 + j])) > big) big = temp;
      }
      if(big == 0){
	dflag[ie] = 1;
	break;
	//printf("Abort::Singular Matrix!!! - By thread %d\n", ie);
      }
      vv[i] = 1/big;
    }

    if(dflag[ie] == 0){
      for (int j = 0; j < 6; ++j){	  //Loop over columns of  Crout's method
	for (int i = 0; i < j; ++i){
	  sum = a[i*6+j];
	  for (int k = 0; k < i; ++k) sum -= a[i*6+k]*a[k*6+j];
	  a[i*6+j] = sum;
	}
	//Search for largest pivot element
	big = 0.0;
	for (int i = j; i < 6; ++i){
	  sum = a[i*6+j];
	  for (int k = 0; k < j; ++k) sum -= a[i*6+k]*a[k*6+j];
	  a[i*6+j] = sum;
	  if ( (dum=vv[i]*fabs(sum))>=big){
	    big = dum;
	    imax = i;
	  }
	}

	if (j != imax){			// Do we need to interchange rows?
	  for (int k = 0; k < 6; ++k){	//Yes
	    dum = a[imax*6+k];
	    a[imax*6+k] = a[j*6+k];
	    a[j*6+k] = dum;
	  }
	  //d = -d;				//change the parity of d;
	  vv[imax] = vv[j];	//Interchange the scale factor;
	}
      
	idx[j] = imax;
	if (a[j*6+j]==0)	a[j*6+j] = 1.0e-20;	//If the pivot element is zero, submitted by a tiny value
		
	if(j != 5){	    //Divide by the pivot element
	  dum = 1.0/(a[j*6+j]);
	  for (int i=j+1; i<6; ++i) a[i*6+j] *= dum;
	}
      }
    
    
      //lubksb
      int ii=-1, ip;
  
      for(int i=0; i < 6; ++i){
	ip = idx[i];
	sum = f[ip*Npart + ie];
	f[ip*Npart + ie] = f[i*Npart + ie];
	if(ii + 1)
	  for(int j = ii; j <= i-1; ++j)	sum -= a[i*6+j]*f[j*Npart + ie];
	else if (sum) ii=i;
	f[i*Npart + ie] = sum; 
      }
    

      for (int i=5; i>=0; --i){
	sum = f[i*Npart + ie];
	for(int j=i+1;j < 6; ++j)	{
	  sum -= a[i*6+j]*f[j*Npart + ie];
	}
      
	f[i*Npart + ie] = sum/a[i*6+i];
      }
    
      for(int i = 0; i < 6; ++i){
	xf[i * Npart + ie] -= f[i * Npart + ie];
      }
    }
  }
}

//1e-16	//Tolerance for Newton solver
//flag == 0 : Particle requires further computation
__global__
void
checkResidual(int *outOfBound, double *f, int *dflag, double epsabs, int Npart){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double residual = 0;
  if(ie < Npart){
    for(int i = 0; i< 6; ++i){
      residual += fabs(f[i * Npart + ie]);
    }
    //if(ie <= 3)
    //printf("%d\t%.15e\t%.15e %d\n",ie,residual, epsabs, residual < epsabs);
    int flag = 0;
    if(residual < epsabs || outOfBound[ie] == 1)
      flag = 1;
    dflag[ie] = flag;
  }
}



#include "SympTr.cu"

