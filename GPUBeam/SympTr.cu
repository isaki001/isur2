#include <iostream>
#include <iomanip>

using namespace std;


__global__
void
gf2Eqns(int const* __restrict__ itM, double *xi, double *xf, double *f_xi_xf, int Npart, int maxLen, int iNrow, int attribId){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
  double y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0, y5 = 0;
  double z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0;
  double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0;
  
  if(ie < Npart/PARTICLES_PER_THREAD){
    x0 = xi[ie];
    x1 = xf[Npart + ie];
    x2 = xi[Npart * 2 + ie];
    x3 = xf[Npart * 3 + ie];
    x4 = xi[Npart * 4 + ie];
    x5 = xf[Npart * 5 + ie];
    
    ie += Npart/PARTICLES_PER_THREAD;
    y0 = xi[ie];
    y1 = xf[Npart + ie];
    y2 = xi[Npart * 2 + ie];
    y3 = xf[Npart * 3 + ie];
    y4 = xi[Npart * 4 + ie];
    y5 = xf[Npart * 5 + ie];    

    
    ie += Npart/PARTICLES_PER_THREAD;
    z0 = xi[ie];
    z1 = xf[Npart + ie];
    z2 = xi[Npart * 2 + ie];
    z3 = xf[Npart * 3 + ie];
    z4 = xi[Npart * 4 + ie];
    z5 = xf[Npart * 5 + ie];    

    ie += Npart/PARTICLES_PER_THREAD;
    w0 = xi[ie];
    w1 = xf[Npart + ie];
    w2 = xi[Npart * 2 + ie];
    w3 = xf[Npart * 3 + ie];
    w4 = xi[Npart * 4 + ie];
    w5 = xf[Npart * 5 + ie];        
  }

  applyMapByAtrib(itM, /*x, op,*/ 
		  x0, x1, x2, x3, x4, x5,
		  y0, y1, y2, y3, y4, y5,
		  z0, z1, z2, z3, z4, z5,
		  w0, w1, w2, w3, w4, w5,
		  Npart, iNrow, maxLen, attribId);

  ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  if(ie < Npart/PARTICLES_PER_THREAD && (attribId%2 == 0) ){
    x0 = xf[attribId * Npart + ie] - x0;
    f_xi_xf[attribId * Npart + ie] = x0;

    y0 = xf[attribId * Npart + Npart/PARTICLES_PER_THREAD + ie] - y0;
    f_xi_xf[attribId * Npart + Npart/PARTICLES_PER_THREAD + ie] = y0;

    z0 = xf[attribId * Npart + 2*Npart/PARTICLES_PER_THREAD + ie] - z0;
    f_xi_xf[attribId * Npart + 2*Npart/PARTICLES_PER_THREAD + ie] = z0;

    w0 = xf[attribId * Npart + 3*Npart/PARTICLES_PER_THREAD + ie] - w0;
    f_xi_xf[attribId * Npart + 3*Npart/PARTICLES_PER_THREAD + ie] = w0;
  }else if(ie < Npart/PARTICLES_PER_THREAD && (attribId%2 == 1) ){
    x0 = xi[attribId * Npart + ie] - x0;
    f_xi_xf[attribId * Npart + ie] = x0;

    y0 = xi[attribId * Npart + Npart/PARTICLES_PER_THREAD + ie] - y0;
    f_xi_xf[attribId * Npart + Npart/PARTICLES_PER_THREAD + ie] = y0;

    z0 = xi[attribId * Npart + 2*Npart/PARTICLES_PER_THREAD + ie] - z0;
    f_xi_xf[attribId * Npart + 2*Npart/PARTICLES_PER_THREAD + ie] = z0;

    w0 = xi[attribId * Npart + 3*Npart/PARTICLES_PER_THREAD + ie] - w0;
    f_xi_xf[attribId * Npart + 3*Npart/PARTICLES_PER_THREAD + ie] = w0;
  }
}

__global__
void
JacobianRow0(int const* __restrict__ itM, 
	 double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
	     int Npart, int maxLen,	int iNrow, int attribId, int rowIdx){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;  
  double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
  double y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0, y5 = 0;
  double z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0;
  double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0;
  
  if(ie < Npart/PARTICLES_PER_THREAD){
    if(dflag[ie] == 0){
      x0 = xi[ie];
      x1 = xf[Npart + ie];
      x2 = xi[Npart * 2 + ie];
      x3 = xf[Npart * 3 + ie];
      x4 = xi[Npart * 4 + ie];
      x5 = xf[Npart * 5 + ie];
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      y0 = xi[ie];
      y1 = xf[Npart + ie];
      y2 = xi[Npart * 2 + ie];
      y3 = xf[Npart * 3 + ie];
      y4 = xi[Npart * 4 + ie];
      y5 = xf[Npart * 5 + ie];    
    }
    
    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      z0 = xi[ie];
      z1 = xf[Npart + ie];
      z2 = xi[Npart * 2 + ie];
      z3 = xf[Npart * 3 + ie];
      z4 = xi[Npart * 4 + ie];
      z5 = xf[Npart * 5 + ie];    
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      w0 = xi[ie];
      w1 = xf[Npart + ie];
      w2 = xi[Npart * 2 + ie];
      w3 = xf[Npart * 3 + ie];
      w4 = xi[Npart * 4 + ie];
      w5 = xf[Npart * 5 + ie];        
    }
  }  



  ie = blockIdx.x * BLOCKDIMX + threadIdx.x;

  applyMapByAtrib(itM, /*x, op,*/ 
		  x0, x1, x2, x3, x4, x5,
		  y0, y1, y2, y3, y4, y5,
		  z0, z1, z2, z3, z4, z5,
		  w0, w1, w2, w3, w4, w5,
		  Npart, iNrow, maxLen, attribId);


  ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  if(ie < Npart/PARTICLES_PER_THREAD){
    if(dflag[ie] == 0 ){    
      double f = xf[rowIdx * Npart + ie];
      double delta = sqrt(PRECISION*fabs(f));
      if (delta==0) delta = PRECISION;

      if(attribId == rowIdx){
	f += delta;
	x0 = f - x0;
      }else if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	x0 = x - x0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	x0 = x - x0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (x0 - f_xi_xf[attribId * Npart + ie])/delta; 
    }
    
    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      double f = xf[rowIdx * Npart + ie];
      double delta = sqrt(PRECISION*fabs(f));
      if (delta==0) delta = PRECISION;

      if(attribId == rowIdx){
	f += delta;
	y0 = f - y0;
      }else if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	y0 = x - y0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	y0 = x - y0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (y0 - f_xi_xf[attribId * Npart + ie])/delta; 
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      double f = xf[rowIdx * Npart + ie];
      double delta = sqrt(PRECISION*fabs(f));
      if (delta==0) delta = PRECISION;

      if(attribId == rowIdx){
	f += delta;
	z0 = f - z0;
      }else if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	z0 = x - z0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	z0 = x - z0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (z0 - f_xi_xf[attribId * Npart + ie])/delta; 
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      double f = xf[rowIdx * Npart + ie];
      double delta = sqrt(PRECISION*fabs(f));
      if (delta==0) delta = PRECISION;

      if(attribId == rowIdx){
	f += delta;
	w0 = f - w0;
      }else if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	w0 = x - w0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	w0 = x - w0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (w0 - f_xi_xf[attribId * Npart + ie])/delta; 
    }
  }  
}




__global__
void
JacobianRow1(int const* __restrict__ itM, 
	     double *xi, double *xf, double *f_xi_xf, double *jac, int *dflag,
	     int Npart, int maxLen,	int iNrow, int attribId, int rowIdx){
  int ie = blockIdx.x * BLOCKDIMX + threadIdx.x;  
  double x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0, deltax = 0;
  double y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0, y5 = 0, deltay = 0;
  double z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0, deltaz = 0;
  double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, deltaw = 0;
  
  if(ie < Npart/PARTICLES_PER_THREAD){
    if(dflag[ie] == 0){
      x0 = xi[ie];
      x1 = xf[Npart + ie];
      x2 = xi[Npart * 2 + ie];
      x3 = xf[Npart * 3 + ie];
      x4 = xi[Npart * 4 + ie];
      x5 = xf[Npart * 5 + ie];
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      y0 = xi[ie];
      y1 = xf[Npart + ie];
      y2 = xi[Npart * 2 + ie];
      y3 = xf[Npart * 3 + ie];
      y4 = xi[Npart * 4 + ie];
      y5 = xf[Npart * 5 + ie];    
    }
    
    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      z0 = xi[ie];
      z1 = xf[Npart + ie];
      z2 = xi[Npart * 2 + ie];
      z3 = xf[Npart * 3 + ie];
      z4 = xi[Npart * 4 + ie];
      z5 = xf[Npart * 5 + ie];    
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      w0 = xi[ie];
      w1 = xf[Npart + ie];
      w2 = xi[Npart * 2 + ie];
      w3 = xf[Npart * 3 + ie];
      w4 = xi[Npart * 4 + ie];
      w5 = xf[Npart * 5 + ie];        
    }

    if(rowIdx == 1){
      deltax = sqrt(PRECISION*fabs(x1));
      if (deltax==0) deltax = PRECISION;
      x1 += deltax;

      deltay = sqrt(PRECISION*fabs(y1));
      if (deltay==0) deltay = PRECISION;
      y1 += deltay;

      deltaz = sqrt(PRECISION*fabs(z1));
      if (deltaz == 0) deltaz = PRECISION;
      z1 += deltaz;

      deltaw = sqrt(PRECISION*fabs(w1));
      if (deltaw == 0) deltaw = PRECISION;
      w1 += deltaw;
    }else if(rowIdx == 3){
      deltax = sqrt(PRECISION*fabs(x3));
      if (deltax==0) deltax = PRECISION;
      x3 += deltax;

      deltay = sqrt(PRECISION*fabs(y3));
      if (deltay==0) deltay = PRECISION;
      y3 += deltay;

      deltaz = sqrt(PRECISION*fabs(z3));
      if (deltaz==0) deltaz = PRECISION;
      z3 += deltaz;

      deltaw = sqrt(PRECISION*fabs(w3));
      if (deltaw==0) deltaw = PRECISION;
      w3 += deltaw;
    }else if(rowIdx == 5){
      deltax = sqrt(PRECISION*fabs(x5));
      if (deltax==0) deltax = PRECISION;
      x5 += deltax;

      deltay = sqrt(PRECISION*fabs(y5));
      if (deltay==0) deltay = PRECISION;
      y5 += deltay;

      deltaz = sqrt(PRECISION*fabs(z5));
      if (deltaz==0) deltaz = PRECISION;
      z5 += deltaz;

      deltaw = sqrt(PRECISION*fabs(w5));
      if (deltaw==0) deltaw = PRECISION;
      w5 += deltaw;
    }


  }  

  //ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  applyMapByAtrib(itM, /*x, op,*/ 
		  x0, x1, x2, x3, x4, x5,
		  y0, y1, y2, y3, y4, y5,
		  z0, z1, z2, z3, z4, z5,
		  w0, w1, w2, w3, w4, w5,
		  Npart, iNrow, maxLen, attribId);


  
  ie = blockIdx.x * BLOCKDIMX + threadIdx.x;
  if(ie < Npart/PARTICLES_PER_THREAD){
    if(dflag[ie] == 0 ){    
      if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	x0 = x - x0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	x0 = x - x0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (x0 - f_xi_xf[attribId * Npart + ie])/deltax; 
    }
    
    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	y0 = x - y0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	y0 = x - y0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (y0 - f_xi_xf[attribId * Npart + ie])/deltay; 
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	z0 = x - z0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	z0 = x - z0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (z0 - f_xi_xf[attribId * Npart + ie])/deltaz; 
    }

    ie += Npart/PARTICLES_PER_THREAD;
    if(dflag[ie] == 0){
      if(attribId%2 == 0){
	double x = xf[attribId * Npart + ie];
	w0 = x - w0;
      }else if(attribId%2 == 1){
	double x = xi[attribId * Npart + ie];
	w0 = x - w0;
      }
      jac[rowIdx * 6 * Npart + attribId * Npart + ie] = (w0 - f_xi_xf[attribId * Npart + ie])/deltaw; 
    }
  }  
}
//Check residual is per particle.. performance residual check if a particle needs to be altered...
//Mistake - even if one particle was flagged you corrected all the particles, which is wrong!
void
NewtonIter1(int *dOutOfBound, int *itM, double *xi, double *xf, int Npart, int maxLen,
	    int iNrow0, int iNrow1, int iNrow2, int iNrow3, int iNrow4, int iNrow5){
  
  int aNrow[6];
  aNrow[0] = iNrow0;
  aNrow[1] = iNrow1;
  aNrow[2] = iNrow2;
  aNrow[3] = iNrow3;
  aNrow[4] = iNrow4;
  aNrow[5] = iNrow5;
  
  double *f_xi_xf = 0;
  cudaMalloc((void**)&f_xi_xf, Npart * 6 *sizeof(double));
  int numThreads = BLOCKDIMX;
  
  cudaStream_t stream[NCOL];
  for (int i0 = 0; i0 < NCOL; i0++) {
    QuadDebug(cudaStreamCreate(&stream[i0]));
  }
  
  
  int numReqdThreads = Npart/PARTICLES_PER_THREAD;
  int numBlocks = numReqdThreads/numThreads + ((numReqdThreads%numThreads)?1:0);
  gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[0]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow0, 0);  
  gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[1]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow1, 1);
  gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[2]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow2, 2);
  gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[3]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow3, 3);
  gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[4]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow4, 4);
  gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[5]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow5, 5);

  for(int i0 = 0; i0  < NCOL; i0++) {
    cudaStreamSynchronize(stream[i0]);
  }
  cudaDeviceSynchronize();

  double *jac = 0;
  cudaMalloc((void **)&jac, Npart * 36 * sizeof(double));
  cudaMemset(jac, 0, Npart * 36 * sizeof(double));  
  int *dflag = 0;
  cudaMalloc((void **)&dflag, Npart * sizeof(int));

  int nIter = 4;
  for(int i = 0; i < nIter; ++i){    
    double epsabs = 1e-12;
    if(epsabs < 0.0){
      std::cout << "\nAbsolute tolerence is negative!!!\n" << std::endl;
      exit(EXIT_FAILURE);
    }
    int numBlocks = Npart/numThreads + ((Npart%numThreads)?1:0);
    checkResidual<<<numBlocks, numThreads, 0, stream[0]>>>(dOutOfBound, f_xi_xf, dflag, epsabs, Npart);
    cudaStreamSynchronize(stream[0]);
    cudaDeviceSynchronize();
    
    //TODO -- Check the condition of flag
    thrust::device_ptr<int> ptr;
    ptr = thrust::device_pointer_cast(dflag);
    int numValidParticles = thrust::reduce(ptr, ptr + Npart, 0, thrust::plus<int>());
    //cudaDeviceSynchronize();
    
    if(numValidParticles==Npart){
      //Cleaning up takes place after the break
      //cudaFree(f_xi_xf);
      //cudaFree(dflag);
      //cudaFree(jac);
      break;
    }
    
    
    for(int ii = 0; ii < NCOL ;++ii){
      int numReqdThreads = Npart/PARTICLES_PER_THREAD;
      int numBlocks = numReqdThreads/numThreads + ((numReqdThreads%numThreads)?1:0);

      JacobianRow0<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[0]>>>(itM, xi, xf, f_xi_xf, jac, dflag, Npart, maxLen, aNrow[ii], ii, 0);
      JacobianRow0<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[1]>>>(itM, xi, xf, f_xi_xf, jac, dflag, Npart, maxLen, aNrow[ii], ii, 2);
      JacobianRow0<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[2]>>>(itM, xi, xf, f_xi_xf, jac, dflag, Npart, maxLen, aNrow[ii], ii, 4);

      JacobianRow1<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[3]>>>(itM, xi, xf, f_xi_xf, jac, dflag, Npart, maxLen, aNrow[ii], ii, 1);
      JacobianRow1<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[4]>>>(itM, xi, xf, f_xi_xf, jac, dflag, Npart, maxLen, aNrow[ii], ii, 3);
      JacobianRow1<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[5]>>>(itM, xi, xf, f_xi_xf, jac, dflag, Npart, maxLen, aNrow[ii], ii, 5);

      for(int i0 = 0; i0  < NCOL; i0++) {
	cudaStreamSynchronize(stream[i0]);
      }
    }
    cudaDeviceSynchronize();
        
    numBlocks = Npart/numThreads + ((Npart%numThreads)?1:0);
    LUDcmp_lubksb<<<numBlocks, numThreads, 0, stream[0]>>>(jac, f_xi_xf, xf, dflag, Npart);
    cudaStreamSynchronize(stream[0]);
    cudaDeviceSynchronize();
    
    {
      int numReqdThreads = Npart/PARTICLES_PER_THREAD;
      int numBlocks = numReqdThreads/numThreads + ((numReqdThreads%numThreads)?1:0);
      gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[0]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow0, 0);  
      gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[1]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow1, 1);
      gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[2]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow2, 2);
      gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[3]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow3, 3);
      gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[4]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow4, 4);
      gf2Eqns<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[5]>>>(itM, xi, xf, f_xi_xf, Npart, maxLen, iNrow5, 5);

      for(int i0 = 0; i0  < NCOL; i0++) {
	cudaStreamSynchronize(stream[i0]);
      }
    }
    cudaDeviceSynchronize();
  }

  for(int i0 = 0; i0 < NCOL; i0++)
    QuadDebug(cudaStreamDestroy(stream[i0]));

  cudaFree(f_xi_xf);
  cudaFree(dflag);
  cudaFree(jac);
}


double
applyMapGPU(int *&dOutOfBound,
	    int *&itM, int *eqnsitM, 
	    double *&dx, double *&dOpx,
	    int *&Nrow, int *&eqnsNrow,
	    /*Map *&map, Map *&eqns,*/ 
	    double *&x, 
	    int maxLen, int maxLen_eqns,
	    int Npart, int Ncol, BeamParams *bParams, int iTurn){
  int numThreads = BLOCKDIMX;

  cudaStream_t stream[Ncol];
  for (int i0 = 0; i0 < Ncol; i0++) {
    QuadDebug(cudaStreamCreate(&stream[i0]));
  }

  quad::timer::event_pair timer0;
  quad::timer::start_timer(&timer0);
  
  for(int i = 0; i < 6; ++i){
    int numReqdThreads = Npart/PARTICLES_PER_THREAD;
    int numBlocks = numReqdThreads/numThreads + ((numReqdThreads%numThreads)?1:0);
    //cout<<"blocks:"<<numBlocks<<endl;
	applyMSHFL<<<numBlocks, numThreads, sizeof(int) * numThreads, stream[i]>>>(dOutOfBound, itM, dx, dOpx, Npart, maxLen, Nrow[i], i, bParams->x_bound, bParams->y_bound);
  }
  for(int i0 = 0; i0 < Ncol; i0++)
    QuadDebug(cudaStreamSynchronize(stream[i0]));

  if(bParams->isSymTr && bParams->NSympFreq > 0 && (iTurn % bParams->NSympFreq == 0)){
    NewtonIter1(dOutOfBound, eqnsitM, dx, dOpx, Npart, maxLen_eqns, eqnsNrow[0], eqnsNrow[1], eqnsNrow[2], eqnsNrow[3], eqnsNrow[4], eqnsNrow[5]);
  }
  double time = quad::timer::stop_timer_returntime(&timer0, "GPU applyM");

  for(int i0 = 0; i0 < Ncol; i0++)
    QuadDebug(cudaStreamDestroy(stream[i0]));

  QuadDebug(cudaMemcpy(dx, dOpx, sizeof(double) * (Ncol + 2) * Npart, cudaMemcpyDeviceToDevice));
  return time;
}
