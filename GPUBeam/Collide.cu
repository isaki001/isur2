/*
#Include <iostream>
#include <iomanip>
#include "Collide.h"
#include <cuComplex.h>
#include "Util.cu"
#include <complex.h>
#include "../util/Util.h"
#include <cuda_runtime.h>
#include "../util/cudaArchUtil.h"
#include "../util/cudaDebugUtil.h"
#include "../util/cudaUtil.h"
#include "../util/cudaTimerUtil.h"
#include "../util/cudaMemoryUtil.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define BLOCKDIMX 128
*/

//#include "../Simulator.h"
//#include <cuda_runtime.h>
//#include "../util/cudaArchUtil.h"
//#include "../util/cudaDebugUtil.h"
//#include "../util/cudaUtil.h"
//#include "../util/cudaTimerUtil.h"
//#include "../util/cudaMemoryUtil.h"
//#include <mpi.h>
#include <iomanip>
#include "Collide.h"
//#include "../util/Util.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <cuComplex.h>
#include "Util.cu"
#include <complex.h>
#include <mpi.h>

#define BLOCKDIMX 128

using namespace std;

//__constant__ int c_N;

//!-------------------
//! STRONG-STRONG MODE
//!------------------------------------------------------------------
/*
__device__
void
pRegimeStrongStrong(double &x, double &px, double &y, double &py,
		    double *dx_e, double *dxbar_e, double *dybar_e, double *dsig_x_e, double *dsig_y_e, int *dSe, double gamma_e, int Npart_e, double N_e,
		    double *dx_p, double *dxbar_p, double *dybar_p, double *dsig_x_p, double *dsig_y_p, int *dSp, double gamma_p, int Npart_p, double N_p,
		    double s, int curSlices, int sliceId, int Npart_inbound_e, int Npart_inbound_p){
  cuDoubleComplex eye = make_cuDoubleComplex(0.0 , 1.0);
  cuDoubleComplex z1, z2, w1, w2, fk;
  
//  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  double xbar_e = dxbar_e[(curSlices - 1) - sliceId];
  double ybar_e = dybar_e[(curSlices - 1) - sliceId];

  double sig_x_e = dsig_x_e[(curSlices - 1) - sliceId];
  double sig_y_e = dsig_y_e[(curSlices - 1) - sliceId];
  int iSe = dSe[(curSlices - 1) - sliceId];

  double sig_x_p = dsig_x_p[sliceId];
  double sig_y_p = dsig_y_p[sliceId];
  int iSp = dSp[sliceId];

  double sxy_e = 0;
  double syx_e = 0;

  double sg_e = 0;
  double sg_p = 0;

  if(iSe > 1) {
  if( sig_x_e != 0 && sig_y_e != 0)
  {
  sxy_e = sig_x_e/sig_y_e;
  syx_e = sig_y_e/sig_x_e;
  }
  }

  if(iSe > 1) {
  sg_e = 1.0/sqrt(2.0 * (fabs(sig_x_e * sig_x_e - sig_y_e * sig_y_e)));
  }
  if(iSp > 1) {
  sg_p = 1.0/sqrt(2.0 * (fabs(sig_x_p * sig_x_p - sig_y_p * sig_y_p)));
  }


  double x1 = x + s * px - xbar_e;
  double y1 = y + s * py - ybar_e;

  double eterm = exp(-0.50 * pow((x1/sig_x_e), 2.0) - 0.50 * pow((y1/sig_y_e), 2.0));
  double fcnst_e = (iSp*Re/gamma_e)*(2.0*sqrt(PI)*sg_p)*(N_p/Npart_inbound_p);
  double fcnst_p = (iSe*Rp/gamma_p)*(2.0*sqrt(PI)*sg_e)*(N_e/Npart_inbound_e);

  x1 = x1*sg_e;
  y1 = y1*sg_e;

  double x2 = syx_e*x1;
  double y2 = sxy_e*y1;
  double Fx = 0, Fy = 0;
  
  if (sig_x_e > sig_y_e){
    z1 = make_cuDoubleComplex(x1 + cuCreal(eye) * fabs(y1), cuCimag(eye)* fabs(y1));
    z2 = make_cuDoubleComplex(x2 + cuCreal(eye) * fabs(y2), cuCimag(eye)* fabs(y2));
	//printf(" tid = %d     z1 = %e, %e \n", tid, cuCreal(z1), cuCimag(z1)); 
    w1 = WOFZ(cuCreal(z1), cuCimag(z1));
    w2 = WOFZ(cuCreal(z2), cuCimag(z2));
        
    cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
    fk = make_cuDoubleComplex(fcnst_p*cuCreal(tempC), fcnst_p*cuCimag(tempC));

    if (y1 > 0.0){
      Fx = cuCimag(fk);
      Fy = cuCreal(fk);
    }else{
      Fx = cuCimag(fk);
      Fy = -cuCreal(fk);
    }
  }else{
    z1 = make_cuDoubleComplex(y1 + cuCreal(eye) * fabs(x1), cuCimag(eye)* fabs(x1));
    z2 = make_cuDoubleComplex(y2 + cuCreal(eye) * fabs(x2), cuCimag(eye)* fabs(x2));
        //printf(" tid = %d    z1 = %e, %e \n", tid, cuCreal(z1), cuCimag(z1));


    w1 = WOFZ(cuCreal(z1), cuCimag(z1));
    w2 = WOFZ(cuCreal(z2), cuCimag(z2));
    
    cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
    //fk = make_cuDoubleComplex(fcnst_e*cuCreal(tempC), fcnst_e*cuCimag(tempC));
	    fk = make_cuDoubleComplex(fcnst_p*cuCreal(tempC), fcnst_p*cuCimag(tempC));
    if (x1 < 0.0){
      Fx = -cuCreal(fk);
      Fy = cuCimag(fk);
    }else{
      Fx = cuCreal(fk);
      Fy = cuCimag(fk);
    }    
  }
         //printf(" tid = %d    Fx = %e \n", tid, Fx);
 
  
  x = x + s * Fx;
  px = px - Fx;
  y = y + s * Fy;
  py = py - Fy;

         //printf(" tid = %d    x = %e \n", tid, x);

  

}

//!-------------------
//! STRONG-STRONG MODE
//!------------------------------------------------------------------
__device__
void
eRegimeStrongStrong(double &x, double &px, double &y, double &py,
		    double *dx_e, double *dxbar_e, double *dybar_e, double *dsig_x_e, double *dsig_y_e, int *dSe, double gamma_e, int Npart_e, double N_e,
		    double *dx_p, double *dxbar_p, double *dybar_p, double *dsig_x_p, double *dsig_y_p, int *dSp, double gamma_p, int Npart_p, double N_p,
		    double s, int curSlices, int sliceId, int Npart_inbound_e, int Npart_inbound_p){
  cuDoubleComplex eye = make_cuDoubleComplex(0.0 , 1.0);
  cuDoubleComplex z1, z2, w1, w2, fk;
  
  //int tid = blockIdx.x * blockDim.x + threadIdx.x;

  double sig_x_e = dsig_x_e[sliceId];
  double sig_y_e = dsig_y_e[sliceId];
  int iSe = dSe[sliceId];

  double xbar_p = dxbar_p[(curSlices - 1) - sliceId];
  double ybar_p = dybar_p[(curSlices - 1) - sliceId];
  double sig_x_p = dsig_x_p[(curSlices - 1) - sliceId];
  double sig_y_p = dsig_y_p[(curSlices - 1) - sliceId];
  int iSp = dSp[(curSlices - 1) - sliceId];

  double sxy_p = 0;
  double syx_p = 0;

  double sg_e = 0;
  double sg_p = 0;

  if(iSp > 1) {
  if( sig_x_p != 0 && sig_y_p != 0)
  {
  sxy_p = sig_x_p/sig_y_p;
  syx_p = sig_y_p/sig_x_p;
  }
  }

  if(iSe > 1) {
  sg_e = 1.0/sqrt(2.0 * (fabs(sig_x_e * sig_x_e - sig_y_e * sig_y_e)));
  }
  if(iSp > 1) {
  sg_p = 1.0/sqrt(2.0 * (fabs(sig_x_p * sig_x_p - sig_y_p * sig_y_p)));
  }


  double fcnst_e = (iSp*Re/gamma_e)*(2.0*sqrt(PI)*sg_p)*(N_p/Npart_inbound_p);
  double fcnst_p = (iSe*Rp/gamma_p)*(2.0*sqrt(PI)*sg_e)*(N_e/Npart_inbound_e);

  double x1 = x - s * px - xbar_p;
  double y1 = y - s * py - ybar_p;
  double eterm = exp(-0.50*pow((x1/sig_x_p), 2.0) - 0.50 * pow((y1/sig_y_p), 2.0));

  x1 = x1*sg_p;
  y1 = y1*sg_p;

  double x2 = syx_p*x1;
  double y2 = sxy_p*y1;
  double Fx = 0, Fy = 0;

  
  
  if (sig_x_p > sig_y_p){
    z1 = make_cuDoubleComplex(x1 + cuCreal(eye) * fabs(y1), cuCimag(eye)* fabs(y1));
    z2 = make_cuDoubleComplex(x2 + cuCreal(eye) * fabs(y2), cuCimag(eye)* fabs(y2));
        //printf(" tid = %d    z1 = %e, %e \n", tid, cuCreal(z1), cuCimag(z1));
 
    w1 = WOFZ(cuCreal(z1), cuCimag(z1));
    w2 = WOFZ(cuCreal(z2), cuCimag(z2));
    
    
    cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
    fk = make_cuDoubleComplex(fcnst_e*cuCreal(tempC), fcnst_e*cuCimag(tempC));

     if (y1 > 0.0){
       Fx = cuCimag(fk);
       Fy = cuCreal(fk);
     }else{
       Fx = cuCimag(fk);
       Fy = -cuCreal(fk);
     }
  }else{
    z1 = make_cuDoubleComplex(y1 + cuCreal(eye) * fabs(x1), cuCimag(eye)* fabs(x1));
    z2 = make_cuDoubleComplex(y2 + cuCreal(eye) * fabs(x2), cuCimag(eye)* fabs(x2));
        //printf(" tid = %d    z1 = %e, %e \n", tid, cuCreal(z1), cuCimag(z1));

    w1 = WOFZ(cuCreal(z1), cuCimag(z1));
    w2 = WOFZ(cuCreal(z2), cuCimag(z2));

    cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
    fk = make_cuDoubleComplex(fcnst_e*cuCreal(tempC), fcnst_e*cuCimag(tempC));
    if (x1 < 0.0){
      Fx = -cuCreal(fk);
      Fy = cuCimag(fk);
    }else{
      Fx = cuCreal(fk);
      Fy = cuCimag(fk);
    }    
  }
  
  
  x = x - s * Fx;
  px = px - Fx;
  y = y - s * Fy;
  py = py - Fy;

}


__global__
void
applyKickGPU(int *dOutOfBound_e, int *dOutOfBound_p, double *dx_e, double *dxbar_e, double *dybar_e, double *dsig_x_e, double *dsig_y_e, int *dSe, double zmin_e, double gamma_e, int Npart_e, double N_e,
	     double *dx_p, double *dxbar_p, double *dybar_p, double *dsig_x_p, double *dsig_y_p, int *dSp, double zmin_p, double gamma_p, int Npart_p, double N_p,
	     double *dS, int numSlices, int curSlices, int iRegime, int isTopTriangle, int Npart_inbound_e, int Npart_inbound_p){
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if(pid < Npart_e && dOutOfBound_e[pid] != 1){
    double L_e    = 2.0 * fabs(zmin_e);  //! make the box size L = 2*|zmin|
    double dz_e  = L_e/(double)numSlices;
    int eSliceId = INT_MAX;

    double z_e  = dx_e[4 * Npart_e + pid];

    eSliceId  = (z_e - zmin_e)/dz_e + 1;
    eSliceId  = MIN(MAX(1,eSliceId), numSlices); //! equivalent to: if (iz>N) iz=N; if (iz<1) iz=1
    eSliceId--;
  
    double x  = dx_e[pid];
    double px = dx_e[Npart_e + pid];
    double y  = dx_e[2 * Npart_e + pid];
    double py = dx_e[3 * Npart_e + pid];    

    if(iRegime == 2 && eSliceId < curSlices && isTopTriangle){
      //printf("e %d\t%d\n", pid, eSliceId);
      double s = dS[eSliceId];
      eRegimeStrongStrong(x, px, y, py, 
			  dx_e, dxbar_e, dybar_e, dsig_x_e, dsig_y_e, dSe, gamma_e, Npart_e, N_e,
			  dx_p, dxbar_p, dybar_p, dsig_x_p, dsig_y_p, dSp, gamma_p, Npart_p, N_p,		    
			  s, curSlices, eSliceId, Npart_inbound_e, Npart_inbound_p);
    }

      
    if(iRegime == 2 && ((numSlices - 1 - eSliceId) < curSlices) && !isTopTriangle){
      double s = dS[eSliceId];
      eSliceId = eSliceId - (numSlices - curSlices);
      eRegimeStrongStrong(x, px, y, py, 
			  dx_e, dxbar_e, dybar_e, dsig_x_e, dsig_y_e, dSe, gamma_e, Npart_e, N_e,
			  dx_p, dxbar_p, dybar_p, dsig_x_p, dsig_y_p, dSp, gamma_p, Npart_p, N_p,		    
			  s, curSlices, eSliceId, Npart_inbound_e, Npart_inbound_p);
    }
    dx_e[pid] = x;
    dx_e[Npart_p + pid] = px;
    dx_e[2 * Npart_p + pid] = y;
    dx_e[3 * Npart_p + pid] = py;
  }

  if(pid < Npart_p && dOutOfBound_p[pid] != 1){
    double L_p    = 2.0 * fabs(zmin_p);  //! make the box size L = 2*|zmin|
    double dz_p  = L_p/(double)numSlices;
    int pSliceId = INT_MAX;

    double z_p  = dx_p[4 * Npart_p + pid];

    pSliceId  = (z_p - zmin_p)/dz_p + 1;
    pSliceId  = MIN(MAX(1,pSliceId), numSlices); //! equivalent to: if (iz>N) iz=N; if (iz<1) iz=1
    pSliceId--;
  

    double x  = dx_p[pid];
    double px = dx_p[Npart_p + pid];
    double y  = dx_p[2 * Npart_p + pid];
    double py = dx_p[3 * Npart_p + pid];    

    //Top Triangle
    if(iRegime == 2 && pSliceId < curSlices && isTopTriangle){
      double s = dS[(curSlices - 1) - pSliceId];
      pRegimeStrongStrong(x, px, y, py, 
			  dx_e, dxbar_e, dybar_e, dsig_x_e, dsig_y_e, dSe, gamma_e, Npart_e, N_e,
			  dx_p, dxbar_p, dybar_p, dsig_x_p, dsig_y_p, dSp, gamma_p, Npart_p, N_p,		    
			  s, curSlices, pSliceId, Npart_inbound_e, Npart_inbound_p);
    }
    //Bottom Triangle
    if(iRegime == 2 && ((numSlices - 1 - pSliceId) < curSlices) && !isTopTriangle){
      double s = dS[(numSlices - 1) - pSliceId + (numSlices - curSlices)]; //Order of S is reverse of e-beam
      pSliceId = pSliceId - (numSlices - curSlices);
      pRegimeStrongStrong(x, px, y, py, 
			  dx_e, dxbar_e, dybar_e, dsig_x_e, dsig_y_e, dSe, gamma_e, Npart_e, N_e,
			  dx_p, dxbar_p, dybar_p, dsig_x_p, dsig_y_p, dSp, gamma_p, Npart_p, N_p,		    
			  s, curSlices, pSliceId, Npart_inbound_e, Npart_inbound_p);
      
    }
    dx_p[pid] = x;
    dx_p[Npart_p + pid] = px;
    dx_p[2 * Npart_p + pid] = y;
    dx_p[3 * Npart_p + pid] = py;  
  }

}


template<typename T>
__device__
void
Reduce(T *data, int N){
  // contiguous range pattern
  for(size_t offset = blockDim.x / 2; offset > 0; offset >>= 1){
    if(threadIdx.x < offset){
      for(int slice = 0; slice < N; ++slice){
	data[slice * blockDim.x + threadIdx.x] += data[slice * blockDim.x + threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  __syncthreads();
}


template<typename T>
__device__
void
init(T *data, int curSlices, T val){
  __syncthreads();
  
  for(int i = 0; i < curSlices; ++i){
    data[i * blockDim.x + threadIdx.x] = 0;
  }
}


//@param eORp Value of 1 implies e-beam and 2 implies p-beam  
__global__
void
sig_xy_reduce(int *dOutOfBound, double *dx, double *S, double *xbar, double *ybar, int Npart, 
	      double *opx, double zmin, int numSlices, int curSlices, int iRegime, int eORp, int isTopTriangle){
  extern __shared__ double sm[];
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  double  L    = 2.0 * fabs(zmin);  //! make the box size L = 2*|zmin|
  double dz    = L/(double)numSlices;
  int sliceId = INT_MAX;
  
  init(sm, curSlices, 0.0);

  double xval = 0, yval = 0;
  bool isParticleActive = false;
  if(pid < Npart && dOutOfBound[pid] != 1){
    double x  = dx[pid];
    double px = dx[Npart + pid];
    double y  = dx[2 * Npart + pid];
    double py = dx[3 * Npart + pid];
    double z  = dx[4 * Npart + pid];

    sliceId  = (z - zmin)/dz + 1;
    sliceId      = MIN(MAX(1,sliceId), numSlices); //! equivalent to: if (iz>N) iz=N; if (iz<1) iz=1
    sliceId--;

    if(iRegime == 2 && sliceId < curSlices && eORp == 1 && isTopTriangle){
      double s = S[sliceId];
      xval = x - s*px - xbar[sliceId];
      yval = y - s*py - ybar[sliceId];
      isParticleActive = true;
    }

    if(iRegime == 2 && ((numSlices - 1 - sliceId) < curSlices) && eORp == 1 && !isTopTriangle){
      double s = S[sliceId];
      sliceId = sliceId - (numSlices - curSlices);
      xval = x - s*px - xbar[sliceId];
      yval = y - s*py - ybar[sliceId];
      isParticleActive = true;
    }
    
    if(iRegime == 2 && sliceId < curSlices && eORp == 2 && isTopTriangle){
      double s = S[(curSlices - 1) - sliceId];
      xval = x + s*px - xbar[sliceId];
      yval = y + s*py - ybar[sliceId];;
      isParticleActive = true;
    }
    if(iRegime == 2 && ((numSlices - 1 - sliceId) < curSlices) && eORp == 2 && !isTopTriangle){
      double s = S[(numSlices - 1) - sliceId + (numSlices - curSlices)]; //Order of S is reverse of e-beam
      sliceId = sliceId - (numSlices - curSlices);
      xval = x + s*px - xbar[sliceId];
      yval = y + s*py - ybar[sliceId];;
      isParticleActive = true;
      
    }
  }
  if(isParticleActive)
    sm[sliceId * blockDim.x + threadIdx.x] = xval * xval;

  __syncthreads();
  Reduce(sm, curSlices);
  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      opx[i * gridDim.x + blockIdx.x] = sm[i * blockDim.x];
    }
  }

  init(sm, curSlices, 0.0);
  if(isParticleActive)
    sm[sliceId * blockDim.x + threadIdx.x] = yval * yval;
  __syncthreads();
  Reduce(sm, curSlices);

  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      opx[(curSlices + i) * gridDim.x + blockIdx.x] = sm[i * blockDim.x];
    }
  }
}

__global__
void
xy_reduce_p(int *dOutOfBound_p, double *dx, double *S, double zmin,
	    int Npart, int numSlices, int curSlices, int iRegime,
	    double *opx, int *activeSlices, int isTopTriangle){
  extern __shared__ double sm[];
  int *intSm = (int *)&sm[0];
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  double  L    = 2.0 * fabs(zmin);  //! make the box size L = 2*|zmin|
  double dz    = L/(double)numSlices;
  int sliceId = INT_MAX;
  
  init(sm, curSlices, 0.0);

  double xval = 0, yval = 0;
  bool isParticleActive = false;
  if(pid < Npart && dOutOfBound_p[pid] != 1){
    double x  = dx[pid];
    double px = dx[Npart + pid];
    double y  = dx[2 * Npart + pid];
    double py = dx[3 * Npart + pid];
    double z  = dx[4 * Npart + pid];

    sliceId  = (z - zmin)/dz + 1;
    sliceId      = MIN(MAX(1,sliceId), numSlices); //! equivalent to: if (iz>N) iz=N; if (iz<1) iz=1
    sliceId--;

    if(iRegime == 2 && sliceId < curSlices && isTopTriangle){
      double s = S[(curSlices - 1) - sliceId]; //Order of S is reverse of e-beam
      xval = x + s*px;
      yval = y + s*py;
      isParticleActive = true;
      //printf("%d %d %.16e\n", pid, sliceId, xval);
    }
    if(iRegime == 2 && ((numSlices - 1 - sliceId) < curSlices) && !isTopTriangle){
      double s = S[(numSlices - 1) - sliceId + (numSlices - curSlices)]; //Order of S is reverse of e-beam
      xval = x + s*px;
      yval = y + s*py;
      isParticleActive = true;
      //if(sliceId == 4)
      //printf("%d %d %.16e\n", pid, sliceId, xval);

      sliceId = sliceId - (numSlices - curSlices);
    }
  }
  if(isParticleActive)
    sm[sliceId * blockDim.x + threadIdx.x] = xval;

  __syncthreads();
  Reduce(sm, curSlices);
  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      opx[i * gridDim.x + blockIdx.x] = sm[i * blockDim.x];
    }
  }

  init(sm, curSlices, 0.0);
  if(isParticleActive)
    sm[sliceId * blockDim.x + threadIdx.x] = yval;
  __syncthreads();
  Reduce(sm, curSlices);

  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      opx[(curSlices + i) * gridDim.x + blockIdx.x] = sm[i * blockDim.x];
    }
  }

  //Count the number of particles in each slice
  init(intSm, curSlices, 0);
  if(isParticleActive){
    intSm[sliceId * blockDim.x + threadIdx.x] = 1;
  } 
  __syncthreads();
  Reduce(intSm, curSlices);
  
  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      activeSlices[i * gridDim.x + blockIdx.x] = intSm[i * blockDim.x];
    }
  }
}
__global__
void
xy_reduce_e(int *dOutOfBound_e, double *dx, double *S, double zmin,
	    int Npart, int numSlices, int curSlices, int iRegime,
	    double *opx, int *activeSlices, int isTopTriangle){
  extern __shared__ double sm[];
  int *intSm = (int *)&sm[0];
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  double  L    = 2.0 * fabs(zmin);  //! make the box size L = 2*|zmin|
  double dz    = L/(double)numSlices;
  int sliceId = INT_MAX;
  
  init(sm, curSlices, 0.0);

  double xval = 0, yval = 0;
  bool isParticleActive = false;
  if(pid < Npart && dOutOfBound_e[pid] != 1){
    double x  = dx[pid];
    double px = dx[Npart + pid];
    double y  = dx[2 * Npart + pid];
    double py = dx[3 * Npart + pid];
    double z  = dx[4 * Npart + pid];

    sliceId  = (z - zmin)/dz + 1;
    sliceId      = MIN(MAX(1,sliceId), numSlices); //! equivalent to: if (iz>N) iz=N; if (iz<1) iz=1
    sliceId--;

    if(iRegime == 2 && sliceId < curSlices && isTopTriangle){   
      double s = S[sliceId];
      xval = x - s*px;
      yval = y - s*py;
      isParticleActive = true;
      //printf("%d %d %.16e\n", pid, sliceId, xval);
    }
    if(iRegime == 2 && ((numSlices - 1 - sliceId) < curSlices) && !isTopTriangle){
      double s = S[sliceId];
      sliceId = sliceId - (numSlices - curSlices);
      xval = x - s*px;
      yval = y - s*py;
      isParticleActive = true;
      //if(sliceId == 3)
      //printf("%d %d %.16e\n", pid, sliceId, xval);
    }
  }
  if(isParticleActive)
    sm[sliceId * blockDim.x + threadIdx.x] = xval;

  __syncthreads();
  Reduce<double>(sm, curSlices);
  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      opx[i * gridDim.x + blockIdx.x] = sm[i * blockDim.x];
    }
  }

  init(sm, curSlices, 0.0);
  if(isParticleActive)
    sm[sliceId * blockDim.x + threadIdx.x] = yval;
  __syncthreads();
  Reduce<double>(sm, curSlices);

  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      opx[(curSlices + i) * gridDim.x + blockIdx.x] = sm[i * blockDim.x];
    }
  }


  //Count the number of particles in each slice
  init(intSm, curSlices, 0);
  if(isParticleActive){
    //printf("%d %d\n", pid, sliceId);
    intSm[sliceId * blockDim.x + threadIdx.x] = 1;
  } 
  __syncthreads();
  Reduce(intSm, curSlices);
  
  if(threadIdx.x == 0){
    for(int i = 0; i < curSlices; ++i){
      activeSlices[i * gridDim.x + blockIdx.x] = intSm[i * blockDim.x];
    }
  }
}


//!----------------------------------------------------------------------------
//! Compute the mean and the standard deviation of the particle distribution 
//! in each of the two transversal coordinates for each beam 
//!----------------------------------------------------------------------------
void computeMeanAndSD(int *dOutOfBound_e, int *dOutOfBound_p, double *dx_e, double *dx_p, double *dS, 
		      int numKicks, double zmin_e, double zmin_p, 
		      double *&hxbar_e, double *&hybar_e, double *&hsig_x_e, double *&hsig_y_e, int *&hiSe,
		      double *&hxbar_p, double *&hybar_p, double *&hsig_x_p, double *&hsig_y_p, int *&hiSp,
		      BeamParams *bParams, int isTopTriangle){
  
  int numThreads = BLOCKDIMX;
  int numBlocks = bParams->Npart_e/numThreads + ((bParams->Npart_e%numThreads)?1:0);

  double *dOpx = 0;
  int *dActiveSlices = 0;
  (cudaMalloc((void **)&dOpx, sizeof(double) * numBlocks * numKicks * 2));
  (cudaMalloc((void **)&dActiveSlices, sizeof(int) * numBlocks * numKicks));

  //Compute xbar_e, ybar_e by reducing x + S*px and y + S*py 

  //printf("\n\n");

	std::cout<<"numThreads is : "<<numThreads<<"\n";
	std::cout<<"numBlocks is : "<<numBlocks<<"\n";
	std::cout<<"numKicks is : "<<numKicks<<"\n";
	std::cout<<"sizeof(double) * numKicks * numThreads is : "<<sizeof(double) * numKicks * numThreads<<"\n"; 

  xy_reduce_e<<<numBlocks, numThreads, sizeof(double) * numKicks * numThreads>>>(dOutOfBound_e, dx_e, dS, zmin_e, bParams->Npart_e, bParams->N, numKicks, bParams->iRegime, dOpx, dActiveSlices, isTopTriangle);

  thrust::device_ptr<int> intPtr;
  thrust::device_ptr<double> dblPtr;
  
  std::cout << std::setprecision(18);
  std::cout << std::scientific;
  for(int i = 0; i < numKicks; ++i){

    intPtr = thrust::device_pointer_cast(dActiveSlices + i * numBlocks);
    hiSe[i] = thrust::reduce(intPtr, intPtr + numBlocks, (int) 0, thrust::plus<int>());
    //std::cout << hiSe[i] << "\n";
    if(hiSe[i] > 1)
    {
    dblPtr = thrust::device_pointer_cast(dOpx + i * numBlocks);    
    hxbar_e[i] = thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0.0, thrust::plus<double>());
    hxbar_e[i] /=(double)hiSe[i];
    //std::cout << hxbar_e[i] << "\n";

    dblPtr = thrust::device_pointer_cast(dOpx + numKicks * numBlocks + i * numBlocks);    
    hybar_e[i] = thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0, thrust::plus<double>())/(double)hiSe[i];
    //std::cout << hybar_e[i] << "\n";
    }
  }

  double *dxbar_e = 0, *dybar_e = 0;
  (cudaMalloc((void **)&dxbar_e, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dybar_e, sizeof(double) * numKicks));
  (cudaMemcpy(dxbar_e, hxbar_e, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dybar_e, hybar_e, sizeof(double) * numKicks, cudaMemcpyHostToDevice));

  //Compute sig_xy by reducing
  sig_xy_reduce<<<numBlocks, numThreads, sizeof(double) * numKicks * numThreads>>>(dOutOfBound_e, dx_e, dS, dxbar_e, dybar_e, bParams->Npart_e, dOpx, zmin_e, bParams->N, numKicks, bParams->iRegime, 1, isTopTriangle);
  
  for(int i = 0; i < numKicks; ++i){
    if(hiSe[i] > 1) 
    {
    dblPtr = thrust::device_pointer_cast(dOpx + i * numBlocks);    
    hsig_x_e[i] = sqrt(thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0, thrust::plus<double>())/(double)hiSe[i]);
    //std::cout << hsig_x_e[i] << "\n";

    dblPtr = thrust::device_pointer_cast(dOpx + numKicks * numBlocks + i * numBlocks);    
    hsig_y_e[i] = sqrt(thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0, thrust::plus<double>())/(double)hiSe[i]);
    //std::cout << hsig_y_e[i] << "\n";
    }
  }

  
  numThreads = BLOCKDIMX;
  numBlocks = bParams->Npart_p/numThreads + ((bParams->Npart_p%numThreads)?1:0);

  (cudaMalloc((void **)&dOpx, sizeof(double) * numBlocks * numKicks * 2));
  (cudaMalloc((void **)&dActiveSlices, sizeof(int) * numBlocks * numKicks));


  //For p-beam
  xy_reduce_p<<<numBlocks, numThreads, sizeof(double) * numKicks * numThreads>>>(dOutOfBound_p, dx_p, dS, zmin_p, bParams->Npart_p, bParams->N, numKicks, bParams->iRegime, dOpx, dActiveSlices, isTopTriangle);

  //display<double>(dOpx,1);
  
  for(int i = 0; i < numKicks; ++i){
    intPtr = thrust::device_pointer_cast(dActiveSlices + i * numBlocks);
    hiSp[i] = thrust::reduce(intPtr, intPtr + numBlocks, (int) 0, thrust::plus<int>());
    //std::cout << hiSp[i] << "\n";
    if(hiSp[i] > 1)
    {
    dblPtr = thrust::device_pointer_cast(dOpx + i * numBlocks);    
    hxbar_p[i] = thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0, thrust::plus<double>())/(double)hiSp[i];
    //std::cout << hxbar_p[i] << "\n";

    dblPtr = thrust::device_pointer_cast(dOpx + numKicks * numBlocks + i * numBlocks);    
    hybar_p[i] = thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0, thrust::plus<double>())/(double)hiSp[i];
    //std::cout << hybar_p[i] << "\n";
    }
  }
  

  double *dxbar_p = 0, *dybar_p = 0;
  (cudaMalloc((void **)&dxbar_p, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dybar_p, sizeof(double) * numKicks));
  (cudaMemcpy(dxbar_p, hxbar_p, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dybar_p, hybar_p, sizeof(double) * numKicks, cudaMemcpyHostToDevice));

  //Compute sig_xy by reducing
  sig_xy_reduce<<<numBlocks, numThreads, sizeof(double) * numKicks * numThreads>>>(dOutOfBound_p, dx_p, dS, dxbar_p, dybar_p, bParams->Npart_p, dOpx, zmin_p, bParams->N, numKicks, bParams->iRegime, 2, isTopTriangle);

  //  display<double>(dOpx, 1);
  
  for(int i = 0; i < numKicks; ++i){
    if(hiSp[i] > 1)
    {
    dblPtr = thrust::device_pointer_cast(dOpx + i * numBlocks);    
    hsig_x_p[i] = sqrt(thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0, thrust::plus<double>())/(double)hiSp[i]);
    //std::cout << hsig_x_p[i] << "\n";

    dblPtr = thrust::device_pointer_cast(dOpx + numKicks * numBlocks + i * numBlocks);    
    hsig_y_p[i] = sqrt(thrust::reduce(dblPtr, dblPtr + numBlocks, (double) 0, thrust::plus<double>())/(double)hiSp[i]);
    //std::cout << hsig_y_p[i] << "\n";
    }
  }
  

  (cudaFree(dActiveSlices));
  (cudaFree(dOpx));
}

void applyKick(int *dOutOfBound_e, int *dOutOfBound_p, double *dx_e, double *dx_p, double *hS, 
	       int numKicks, double zmin_e, double zmin_p, 
	       BeamParams *bParams, int isTopTriangle){

  double *xbar_e = new double[numKicks];
  double *ybar_e = new double[numKicks];
  int *iSe = new int[numKicks];
  double *sig_x_e = new double[numKicks];
  double *sig_y_e = new double[numKicks];

  double *xbar_p = new double[numKicks];
  double *ybar_p = new double[numKicks];
  int *iSp = new int[numKicks];
  double *sig_x_p = new double[numKicks];
  double *sig_y_p = new double[numKicks];

	for(int i=0; i<numKicks; i++)
	{
		xbar_e[i] = 0;
		ybar_e[i] = 0;
		iSe[i] = 0;
		sig_x_e[i] = 0;
		sig_y_e[i] = 0;

		xbar_p[i] = 0;
		ybar_p[i] = 0;
		iSp[i] = 0;
		sig_x_p[i] = 0;
		sig_y_p[i] = 0;
	}

  double *dS = 0;
  (cudaMalloc((void **)&dS, sizeof(double) * bParams->N));
  (cudaMemcpy(dS, hS, sizeof(double) * bParams->N, cudaMemcpyHostToDevice));


  quad::timer::event_pair timer_node;
  quad::timer::start_timer(&timer_node);
  computeMeanAndSD(dOutOfBound_e, dOutOfBound_p, dx_e, dx_p, dS, numKicks, 
		   zmin_e, zmin_p, 
		   xbar_e, ybar_e, sig_x_e, sig_y_e, iSe, 
		   xbar_p, ybar_p, sig_x_p, sig_y_p, iSp, 
		   bParams, isTopTriangle);
  quad::timer::stop_timer(&timer_node, "ComputeMeanAndSD");
  
  std::cout << std::setprecision(18);
  std::cout << std::scientific;

  int nE = 0, nP = 0;
  for(int i = 0; i < numKicks; ++i){
    int ie = i;
    int ip = numKicks - i - 1;
    nE += iSe[ie];
    nP += iSp[ip];
    std::cout << "\n-------------------\n";
    std::cout << "slice\t" << i << "\n";
    std::cout << "eslice\t" << iSe[ie] << "\n";
    std::cout << "pslice\t" << iSp[ip] << "\n";
    std::cout << "xbar_e\t" << xbar_e[ie] << "\n";
    std::cout << "ybar_e\t" << ybar_e[ie] << "\n";
    std::cout << "xbar_p\t" << xbar_p[ip] << "\n";
    std::cout << "ybar_p\t" << ybar_p[ip] << "\n";
    std::cout << "sig_x_e\t" << sig_x_e[ie] << "\n";
    std::cout << "sig_y_e\t" << sig_y_e[ie] << "\n";
    std::cout << "sig_x_p\t" << sig_x_p[ip] << "\n";
    std::cout << "sig_y_p\t" << sig_y_p[ip] << "\n"; 
    
    //std::cout << hS[i] << "\n";
    //std::cout << xbar_e[i] << "\t" << ybar_e[i] << "\n";
    //std::cout << sig_x_e[i] << "\t" << sig_y_e[i] << "\n";
    //std::cout << xbar_p[i] << "\t" << ybar_p[i] << "\n";
    //std::cout << sig_x_p[i] << "\t" << sig_y_p[i] << "\n";
  }

  double *dxbar_e = 0, *dybar_e = 0, *dsig_x_e = 0, *dsig_y_e = 0;
  double *dxbar_p = 0, *dybar_p = 0, *dsig_x_p = 0, *dsig_y_p = 0;
  int *dSe = 0, *dSp = 0;

  (cudaMalloc((void **)&dxbar_e, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dybar_e, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dsig_x_e, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dsig_y_e, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dSe, sizeof(int) * numKicks));
  
  (cudaMalloc((void **)&dxbar_p, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dybar_p, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dsig_x_p, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dsig_y_p, sizeof(double) * numKicks));
  (cudaMalloc((void **)&dSp, sizeof(int) * numKicks));

  (cudaMemcpy(dxbar_e, xbar_e, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dybar_e, ybar_e, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dsig_x_e, sig_x_e, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dsig_y_e, sig_y_e, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dSe, iSe, sizeof(int) * numKicks, cudaMemcpyHostToDevice));

  (cudaMemcpy(dxbar_p, xbar_p, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dybar_p, ybar_p, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dsig_x_p, sig_x_p, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dsig_y_p, sig_y_p, sizeof(double) * numKicks, cudaMemcpyHostToDevice));
  (cudaMemcpy(dSp, iSp, sizeof(int) * numKicks, cudaMemcpyHostToDevice));

  int maxNpart = MAX(bParams->Npart_e, bParams->Npart_p);
  int numThreads = BLOCKDIMX;
  int numBlocks = maxNpart/numThreads + ((maxNpart%numThreads)?1:0);
  
  quad::timer::start_timer(&timer_node);
	applyKickGPU<<<numBlocks, numThreads>>>(dOutOfBound_e, dOutOfBound_p, dx_e, dxbar_e, dybar_e, dsig_x_e, dsig_y_e, dSe, zmin_e, bParams->gamma_e, bParams->Npart_e, bParams->N_e,
					  dx_p, dxbar_p, dybar_p, dsig_x_p, dsig_y_p, dSp, zmin_p, bParams->gamma_p, bParams->Npart_p, bParams->N_p,
					  dS, bParams->N, numKicks, bParams->iRegime, isTopTriangle, bParams->Npart_inbound_e, bParams->Npart_inbound_p);
  quad::timer::stop_timer(&timer_node, "applyKickGPU");
  
  //display<double>(dx_p, bParams->Npart_e);
  //exit(1);
  (cudaFree(dS));
}
*/


__global__
void
find_sigz(double *d_x, double *dx_temp, int Npart, double zbar)
{
  int pid = blockIdx.x * blockDim.x + threadIdx.x;

  if(pid < Npart)
  {
        double z = d_x[4 * Npart + pid];
        dx_temp[pid] = (z - zbar) * (z - zbar);
  }
}

double find_zmin(double *dx, int Npart){
  thrust::device_ptr<double> ptr;
  ptr = thrust::device_pointer_cast(dx);

  //! zmin (boundary) is the outlier
  double zbar = 0.0;
  zbar = thrust::reduce(ptr + 4 * Npart, ptr + 5 * Npart, (double) 0, thrust::plus<double>())/double(Npart);

  double sig_z;
  double *dx_temp;

  cudaMalloc((void **)&dx_temp, sizeof(double) * Npart);
  cudaMemcpy(dx_temp, dx, sizeof(double) * Npart, cudaMemcpyDeviceToDevice);

  int numThreads = BLOCKDIMX;
  int numBlocks = Npart/numThreads + ((Npart%numThreads)?1:0);

  find_sigz<<<numBlocks, numThreads>>>(dx, dx_temp, Npart, zbar);

  ptr = thrust::device_pointer_cast(dx_temp);

  sig_z = thrust::reduce(ptr, ptr + Npart, (double) 0, thrust::plus<double>());

  sig_z = sqrt(sig_z/double(Npart));

  //std::cout << std::setprecision(16);
  //std::cout << "zmin\t:" << zmin << "\n";
  cudaFree(dx_temp);

  return sig_z;   //! make the box size L = 2*|zmin|
}

__global__
void
slice(double *d_x, double zmin, int Npart, int N, int *d_iS, int *dOutOfBound)
{
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(pid < Npart)
  {
        int iz = 0;
        if(dOutOfBound[pid] != 1)
        {
                double L     = 2.0 * fabs(zmin);
                double dz    = L/double(N);
                iz = (d_x[4 * Npart + pid] - zmin)/dz + 1;
                iz = MIN(MAX(1, iz), N);
                iz = iz - 1;

                atomicAdd(&d_iS[iz], 1);
        }
        else
        {
                iz = -1;
        }

        d_x[6 * Npart + pid] = iz;
        d_x[7 * Npart + pid] = pid;

  }

}

__global__
void
sortbeam(double *d_x, double *d_xs, int *d_iS, int *d_iS_Inc, int *d_iS_ext, int Npart, int Npart_ext, int Ncol, double *dS, double *d_xms, double *d_xsd, int N, int EorP)
{
  int pid = blockIdx.x * blockDim.x + threadIdx.x;

  double x, px, y, py, z, pz, S = 0;
  int orig_idx, base_index = 0, base_index_ms = 0, S_Index = 0, Slice_ID = 0, current_idx = 0;
  __shared__ int block_count[NSLICES], actual_index[NSLICES];

  if(pid < Npart)
  {
        Slice_ID = d_x[6 * Npart + pid];

        if(Slice_ID != -1)
        {
                x = d_x[pid];
                px = d_x[Npart + pid];
                y = d_x[2 * Npart + pid];
                py = d_x[3 * Npart + pid];
                z = d_x[4 * Npart + pid];
                pz = d_x[5 * Npart + pid];

                orig_idx = d_x[7 * Npart + pid];


                for(int i = Slice_ID - 1; i >= 0; i--)
                {
                        base_index = base_index + (d_iS_ext[i]);
                }
        }

   }

   if(threadIdx.x == 0) {

        for(int i=0; i < N; i++) {

                block_count[i] = 0;
        }
   }

   __syncthreads();

   if(Slice_ID != -1 && pid < Npart) {
        current_idx = atomicAdd(&block_count[Slice_ID], 1);
        }

   __syncthreads();

   if(threadIdx.x == 0) {

        for(int i = 0; i < N; i++) {

                actual_index[i] = atomicAdd(&d_iS_Inc[i], block_count[i]);
        }

   }
   __syncthreads();

    if(Slice_ID != -1 && pid < Npart) {
        current_idx += actual_index[Slice_ID];
    }

    if(Slice_ID != -1 && pid < Npart){
        d_xs[base_index + current_idx] = x;
        d_xs[base_index + Npart_ext + current_idx] = px;
        d_xs[base_index + (2 * Npart_ext) + current_idx] = y;
        d_xs[base_index + (3 * Npart_ext) + current_idx] = py;
        d_xs[base_index + (4 * Npart_ext) + current_idx] = z;
        d_xs[base_index + (5 * Npart_ext) + current_idx] = pz;
        d_xs[base_index + (6 * Npart_ext) + current_idx] = Slice_ID;
        d_xs[base_index + (7 * Npart_ext) + current_idx] = orig_idx;

        for(int i = Slice_ID - 1; i >= 0; i--)
        {
                base_index_ms = base_index_ms + (d_iS_ext[i]);
        }


        for(int i = Slice_ID; i >= 0; i--)
        {
                S_Index = S_Index + i;
        }

        if(EorP == 1)
        {
                S_Index = S_Index + Slice_ID;

                S = dS[S_Index];

                d_xms[base_index_ms + current_idx] = x - S * px;
                d_xms[base_index_ms + Npart_ext + current_idx] = y - S * py;
                d_xms[base_index_ms + (2 * Npart_ext) + current_idx] = Slice_ID;

                d_xsd[base_index_ms + current_idx] = (x - S * px) * (x - S * px);
                d_xsd[base_index_ms + Npart_ext + current_idx] = (y - S * py) * (y - S * py);
                d_xsd[base_index_ms + (2 * Npart_ext) + current_idx] = Slice_ID;


        }
        else
        {
                S = dS[S_Index];

                d_xms[base_index_ms + current_idx] = x + S * px;
                d_xms[base_index_ms + Npart_ext + current_idx] = y + S * py;
                d_xms[base_index_ms + (2 * Npart_ext) + current_idx] = Slice_ID;

                d_xsd[base_index_ms + current_idx] = (x + S * px) * (x + S * px);
                d_xsd[base_index_ms + Npart_ext + current_idx] = (y + S * py) * (y + S * py);
                d_xsd[base_index_ms + (2 * Npart_ext) + current_idx] = Slice_ID;

        }
        }

}

/*__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
	old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}*/

__global__
void
applyKickGPU_E(double *dx_es, double gamma_e, int Npart_e_ext, double *dxpbar_e, double *dypbar_e, double *dxbar_p, double *dybar_p, double *d_sig_x_p, double *d_sig_y_p, int *d_iSp, double N_p, double *dS, int numKicks, int epart, int Collision_Num, double *dx_emb, double *dx_esb, int topTraingle, int Npart_inbound_p, int N)
{

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
	//if(pid==0)
		//printf("start apply kick\n");
	
	
		
    __shared__ double l_mean_x[NWARPS], l_mean_y[NWARPS], l_sd_x[NWARPS], l_sd_y[NWARPS];
	//printf("apply kick\n");
    if(threadIdx.x == 0) {
        for(int i = 0; i < NWARPS; i++){
            l_mean_x[i] = 0.0;
            l_mean_y[i] = 0.0;
            l_sd_x[i] = 0.0;
            l_sd_y[i] = 0.0;
        }
    }

    __syncthreads();
    
    if(pid < epart){
        int start_index;
	
        if(topTraingle == 1){
            start_index = 0;
        }
        else{
            start_index = Npart_e_ext - epart;
        }

        int eslice_id = dx_es[start_index + (6 * Npart_e_ext) + pid];

        if(eslice_id != -1){
            double x = dx_es[start_index + pid];
			double px = dx_es[start_index + Npart_e_ext + pid];
			double y = dx_es[start_index + (2 * Npart_e_ext) + pid];
			double py = dx_es[start_index + (3 * Npart_e_ext) + pid];

			int pslice_id;

			if(topTraingle == 1){
                pslice_id = numKicks - 1 - eslice_id; // Only valid for top traingle
			}
			else{
				pslice_id = ((N - 1) + Collision_Num) - eslice_id;
			}

			double s;

            if(topTraingle == 1){
                s = dS[eslice_id]; // Only valid for top traingle	
			}
			else{
				s = dS[eslice_id - Collision_Num];
			}
            
            double xbar_p = dxbar_p[pslice_id];
			double ybar_p = dybar_p[pslice_id];

            double x1 = x - s * px - (xbar_p);
            double y1 = y - s * py - (ybar_p);

            double sig_x_p = d_sig_x_p[pslice_id];
            double sig_y_p = d_sig_y_p[pslice_id];

            double eterm = exp(-0.50 * pow((x1/sig_x_p), 2.0) -0.50 * pow((y1/sig_y_p), 2.0));

			double sg_p = 0.0;
            
            int iSp = d_iSp[pslice_id];

			if(iSp > 1)
                sg_p = 1.0/sqrt(2.0*(fabs(sig_x_p*sig_x_p - sig_y_p*sig_y_p)));

			double sxy_p = 0.0;
			double syx_p = 0.0;

            if(iSp > 1) {
                if( sig_x_p != 0.0 && sig_y_p != 0.0){
                    sxy_p = sig_x_p/sig_y_p;
                    syx_p = sig_y_p/sig_x_p;
			     }
            }

            x1 = x1*sg_p;
            y1 = y1*sg_p;

            double x2 = syx_p*x1;
            double y2 = sxy_p*y1;

            double Fx = 0.0, Fy = 0.0;

            double fcnst_e = (iSp*Re/gamma_e)*(2.0*sqrt(PI)*sg_p)*(N_p/Npart_inbound_p);


            cuDoubleComplex eye = make_cuDoubleComplex(0.0 , 1.0);
            cuDoubleComplex z1, z2, w1, w2, fk;


            if(sig_x_p > sig_y_p){
                z1 = make_cuDoubleComplex(x1 + cuCreal(eye) * fabs(y1), cuCimag(eye)* fabs(y1));
                z2 = make_cuDoubleComplex(x2 + cuCreal(eye) * fabs(y2), cuCimag(eye)* fabs(y2));

                w1 = WOFZ(cuCreal(z1), cuCimag(z1));
                w2 = WOFZ(cuCreal(z2), cuCimag(z2));

                cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
                fk = make_cuDoubleComplex(fcnst_e*cuCreal(tempC), fcnst_e*cuCimag(tempC));

                if (y1 > 0.0){
                    Fx = cuCimag(fk);
                    Fy = cuCreal(fk);

                }
                else{
                    Fx = cuCimag(fk);
                    Fy = -cuCreal(fk);
                }
            }

            else{
                z1 = make_cuDoubleComplex(y1 + cuCreal(eye) * fabs(x1), cuCimag(eye)* fabs(x1));
                z2 = make_cuDoubleComplex(y2 + cuCreal(eye) * fabs(x2), cuCimag(eye)* fabs(x2));

                w1 = WOFZ(cuCreal(z1), cuCimag(z1));
                w2 = WOFZ(cuCreal(z2), cuCimag(z2));

                cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
                fk = make_cuDoubleComplex(fcnst_e*cuCreal(tempC), fcnst_e*cuCimag(tempC));

                if (x1 < 0.0){
                    Fx = -cuCreal(fk);
                    Fy = cuCimag(fk);
                }
                else{
                    Fx = cuCreal(fk);
                    Fy = cuCimag(fk);
                }
            }

            x = x - s * Fx;
            px = px - Fx;
            y = y - s * Fy;
            py = py - Fy;

            dx_es[start_index + pid] = x;
            dx_es[start_index + Npart_e_ext + pid] = px;
            dx_es[start_index + (2 * Npart_e_ext) + pid] = y;
            dx_es[start_index + (3 * Npart_e_ext) + pid] = py;

			double s_next = 0.0;

			if(topTraingle == 1){
				if(numKicks == N){
					s_next = dS[eslice_id + numKicks - 1];
				}
				else{
					s_next = dS[eslice_id + numKicks]; // Only valid for top traingle
				}
			}
			else{
				s_next = dS[eslice_id - Collision_Num +  numKicks - 1];
			}

			double x_px = x - s_next * px;
			double y_py = y - s_next * py;

			double x_px_sq = x_px * x_px;
			double y_py_sq = y_py * y_py;

			for (int offset = 16; offset > 0; offset /= 2){
                x_px += __shfl(x_px, threadIdx.x + offset);
                x_px_sq += __shfl(x_px_sq, threadIdx.x + offset);
            }

            for (int offset = 16; offset > 0; offset /= 2){
                y_py += __shfl(y_py, threadIdx.x + offset);
                y_py_sq += __shfl(y_py_sq, threadIdx.x + offset);
            }

            int warpId = threadIdx.x / 32;

            int leadThread = warpId * 32;

			if(threadIdx.x == leadThread){
                l_mean_x[warpId] = x_px;
                l_sd_x[warpId] = x_px_sq;
			}

            if(threadIdx.x == leadThread){
                l_mean_y[warpId] = y_py;			
                l_sd_y[warpId] = y_py_sq;
			}
        }

        __syncthreads();

        if(threadIdx.x == 0){

            double block_mean_x = 0.0;
            double block_mean_y = 0.0;

            for(int i = 0; i < NWARPS; i++){
                block_mean_x += l_mean_x[i];
                block_mean_y += l_mean_y[i];
            }

            dx_emb[(start_index/blockDim.x) + blockIdx.x] = block_mean_x;
            dx_emb[(start_index/blockDim.x) + (Npart_e_ext/blockDim.x) + blockIdx.x] = block_mean_y;

            block_mean_x = 0;
            block_mean_y = 0;

            for(int i = 0; i < NWARPS; i++){
                block_mean_x += l_sd_x[i];
                block_mean_y += l_sd_y[i];
            }

            dx_esb[(start_index/blockDim.x) + blockIdx.x] = block_mean_x;	
            dx_esb[(start_index/blockDim.x) + (Npart_e_ext/blockDim.x) + blockIdx.x] = block_mean_y;
        }
    }
	//if(pid==0)
		//printf("end apply kick\n");
}

__global__
void
applyKickGPU_P(double *dxpbar_p, double *dypbar_p, double *dxbar_e, double *dybar_e, double *d_sig_x_e, double *d_sig_y_e, int *d_iSe, double N_e, double *dx_ps, double gamma_p, int Npart_p_ext, double *dS, int numKicks, int ppart, int Collision_Num, double *dx_pmb, double *dx_psb, int topTraingle, int Npart_inbound_e, int N){

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double l_mean_x[NWARPS], l_mean_y[NWARPS], l_sd_x[NWARPS], l_sd_y[NWARPS];

    if(threadIdx.x == 0) {
        for(int i = 0; i < NWARPS; i++){
            l_mean_x[i] = 0.0;
            l_mean_y[i] = 0.0;
            l_sd_x[i] = 0.0;
            l_sd_y[i] = 0.0;
        }
    }

    __syncthreads();
    
    if(pid < ppart){
        int start_index;

        if(topTraingle == 1){
            start_index = 0;
        }
        else{
            start_index = Npart_p_ext - ppart;
        }

        int pslice_id = dx_ps[start_index + (6 * Npart_p_ext) + pid];

        if(pslice_id != -1){
            double x = dx_ps[start_index + pid];
            double px = dx_ps[start_index + Npart_p_ext + pid];
            double y = dx_ps[start_index + (2 * Npart_p_ext) + pid];
            double py = dx_ps[start_index + (3 * Npart_p_ext) + pid];

			int eslice_id;

            if(topTraingle == 1){
                eslice_id = numKicks - 1 - pslice_id; // Only valid for top traingle
			}
			else{
                eslice_id = ((N - 1) + Collision_Num) - pslice_id;
			}
            
            int iSe = d_iSe[eslice_id];

			double s;

            if(topTraingle == 1){
                s = dS[numKicks - 1 - pslice_id]; // Only valid for top traingle
			}
			else{
				s = dS[eslice_id - Collision_Num];				
			}

            double xbar_e = dxbar_e[eslice_id];
            double ybar_e = dybar_e[eslice_id];

            double x1 = x + s * px - (xbar_e);
            double y1 = y + s * py - (ybar_e);

            double sig_x_e = d_sig_x_e[eslice_id];
            double sig_y_e = d_sig_y_e[eslice_id];

            double eterm = exp(-0.50 * pow((x1/sig_x_e), 2.0) -0.50 * pow((y1/sig_y_e), 2.0));

            double sg_e = 0;

            if(iSe > 1)
                sg_e = 1.0/sqrt(2.0*(fabs(sig_x_e*sig_x_e - sig_y_e*sig_y_e)));

            double sxy_e = 0;
            double syx_e = 0;

            if(iSe > 1){
                if(sig_x_e != 0 && sig_y_e != 0){
                    sxy_e = sig_x_e/sig_y_e;
                    syx_e = sig_y_e/sig_x_e;
                }
            }

            x1 = x1*sg_e;
            y1 = y1*sg_e;

            double x2 = syx_e*x1;
            double y2 = sxy_e*y1;

            double Fx = 0, Fy = 0;

            double fcnst_p = (iSe*Rp/gamma_p)*(2.0*sqrt(PI)*sg_e)*(N_e/Npart_inbound_e);

            cuDoubleComplex eye = make_cuDoubleComplex(0.0 , 1.0);
            cuDoubleComplex z1, z2, w1, w2, fk;


            if(sig_x_e > sig_y_e){
                z1 = make_cuDoubleComplex(x1 + cuCreal(eye) * fabs(y1), cuCimag(eye)* fabs(y1));
                z2 = make_cuDoubleComplex(x2 + cuCreal(eye) * fabs(y2), cuCimag(eye)* fabs(y2));

                w1 = WOFZ(cuCreal(z1), cuCimag(z1));
                w2 = WOFZ(cuCreal(z2), cuCimag(z2));

                cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
                fk = make_cuDoubleComplex(fcnst_p*cuCreal(tempC), fcnst_p*cuCimag(tempC));

                if (y1 > 0.0){
                        Fx = cuCimag(fk);
                        Fy = cuCreal(fk);
                }
                else{
                        Fx = cuCimag(fk);
                        Fy = -cuCreal(fk);
                }
            }

            else{
                z1 = make_cuDoubleComplex(y1 + cuCreal(eye) * fabs(x1), cuCimag(eye)* fabs(x1));
                z2 = make_cuDoubleComplex(y2 + cuCreal(eye) * fabs(x2), cuCimag(eye)* fabs(x2));

                w1 = WOFZ(cuCreal(z1), cuCimag(z1));
                w2 = WOFZ(cuCreal(z2), cuCimag(z2));

                cuDoubleComplex tempC = cuCsub(w1, make_cuDoubleComplex(eterm*cuCreal(w2), eterm*cuCimag(w2)));
                fk = make_cuDoubleComplex(fcnst_p*cuCreal(tempC), fcnst_p*cuCimag(tempC));

                if (x1 < 0.0){
                        Fx = -cuCreal(fk);
                        Fy = cuCimag(fk);
                }
                else{
                        Fx = cuCreal(fk);
                        Fy = cuCimag(fk);
                }
            }

            x = x + s * Fx;
            px = px - Fx;
            y = y + s * Fy;
            py = py - Fy;

            dx_ps[start_index + pid] = x;
            dx_ps[start_index + Npart_p_ext + pid] = px;
            dx_ps[start_index + (2 * Npart_p_ext) + pid] = y;
            dx_ps[start_index + (3 * Npart_p_ext) + pid] = py;

			double s_next = 0;

            if(topTraingle == 1){
                if(numKicks == N){
                    s_next = dS[eslice_id + numKicks];
                }
                else{
                    s_next = dS[eslice_id + numKicks + 1]; // Only valid for top traingle
				}
			}
			else{
				s_next = dS[eslice_id - Collision_Num + numKicks];
			}

            double x_px = x + s_next * px;
            double y_py = y + s_next * py;

            double x_px_sq = x_px * x_px;
            double y_py_sq = y_py * y_py;

            __syncthreads();

            for (int offset = 16; offset > 0; offset /= 2){
                x_px += __shfl(x_px, threadIdx.x + offset);
                x_px_sq += __shfl(x_px_sq, threadIdx.x + offset);
            }

            for (int offset = 16; offset > 0; offset /= 2){
                y_py += __shfl(y_py, threadIdx.x + offset);
                y_py_sq += __shfl(y_py_sq, threadIdx.x + offset);
            }


            int warpId = threadIdx.x / 32;

            int leadThread = warpId * 32;

            if(threadIdx.x == leadThread){
                    l_mean_x[warpId] = x_px;
                    l_sd_x[warpId] = x_px_sq;
            }

            if(threadIdx.x == leadThread){
                    l_mean_y[warpId] = y_py;
                    l_sd_y[warpId] = y_py_sq;
            }

        }

        __syncthreads();

        if(threadIdx.x == 0){
            double block_mean_x = 0;
            double block_mean_y = 0;

            for(int i = 0; i < NWARPS; i++){
                block_mean_x += l_mean_x[i];
                block_mean_y += l_mean_y[i];
            }

            dx_pmb[(start_index/blockDim.x) + blockIdx.x] = block_mean_x;
            dx_pmb[(start_index/blockDim.x) + (Npart_p_ext/blockDim.x) + blockIdx.x] = block_mean_y;

            block_mean_x = 0;
            block_mean_y = 0;

            for(int i = 0; i < NWARPS; i++){
                block_mean_x += l_sd_x[i];
                block_mean_y += l_sd_y[i];
            }

            dx_psb[(start_index/blockDim.x) + blockIdx.x] = block_mean_x;
            dx_psb[(start_index/blockDim.x) + (Npart_p_ext/blockDim.x) + blockIdx.x] = block_mean_y;
        }
	}
}

__global__
void
merge(double *d_x, double *dx_s, int Npart, int Npart_ext)
{
  int pid = blockIdx.x * blockDim.x + threadIdx.x;

  if(pid < Npart_ext)
  {
	int orig_idx = dx_s[(7 * Npart_ext) + pid];

	if(orig_idx != -1)
	{
		d_x[orig_idx] = dx_s[pid];
		d_x[Npart + orig_idx] = dx_s[Npart_ext + pid];
		d_x[2 * Npart + orig_idx] = dx_s[2 * Npart_ext + pid];
		d_x[3 * Npart + orig_idx] = dx_s[3 * Npart_ext + pid];
	}
  }


}


void calculate_rms(int lum_turn, int ebunch_index, int pbunch_index, double *dx_es, double *dx_ps, int N, int *iSe, int *iSp, int *iSe_ext, int *iSp_ext, int Npart_e_ext, int Npart_p_ext, int timestamp) {

    double *xbar_e, *ybar_e, *xbar_p, *ybar_p, *sig_x_e, *sig_y_e, *sig_x_p, *sig_y_p;

    xbar_e = new double[N];
    ybar_e = new double[N];
    xbar_p = new double[N];
    ybar_p = new double[N];

    sig_x_e = new double[N];
    sig_y_e = new double[N];
    sig_x_p = new double[N];
    sig_y_p = new double[N];

    int base_index_ms_e = 0, base_index_ms_p = 0;

    double xbar_e_total = 0, ybar_e_total = 0, xbar_p_total = 0, ybar_p_total = 0, sig_x_e_total = 0, sig_y_e_total = 0, sig_x_p_total = 0, sig_y_p_total = 0;

    thrust::device_ptr<double> ptr;

if(ebunch_index != -1){
    ptr = thrust::device_pointer_cast(dx_es);

    for(int i = 0; i < N; i++)
    {   
        if(iSe[i] > 1)
        {   
            for(int j = i - 1; j >= 0; j--)
            {   
                base_index_ms_e = base_index_ms_e + iSe_ext[j];
            }
            
            
            xbar_e[i] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + iSe[i], (double) 0, thrust::plus<double>())/iSe[i];
            ybar_e[i] = thrust::reduce(base_index_ms_e + ptr + 2 * Npart_e_ext, base_index_ms_e + ptr + 2 * Npart_e_ext + iSe[i], (double) 0, thrust::plus<double>())/iSe[i];
            
            xbar_e_total += xbar_e[i];
            ybar_e_total += ybar_e[i];
            
            base_index_ms_e = 0;
        
        }
        else
        {   
            xbar_e[i] = 0;
            ybar_e[i] = 0;
        }
    }
}

if(pbunch_index != -1){
    ptr = thrust::device_pointer_cast(dx_ps);

    for(int i = 0; i < N; i++)
    {   
        if(iSp[i] > 1)
        {   
            for(int j = i - 1; j >= 0; j--)
            {
                    base_index_ms_p = base_index_ms_p + iSp_ext[j];
            }


            xbar_p[i] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + iSp[i], (double) 0, thrust::plus<double>())/iSp[i];
            ybar_p[i] = thrust::reduce(base_index_ms_p + 2 * Npart_p_ext + ptr, base_index_ms_p +  2 * Npart_p_ext + ptr + iSp[i], (double) 0, thrust::plus<double>())/iSp[i];

            xbar_p_total += xbar_p[i];
            ybar_p_total += ybar_p[i];

            base_index_ms_p = 0;

        }
        else
        {
            xbar_p[i] = 0;
            ybar_p[i] = 0;
        }
    }
}
    double *x_es_temp = new double[Npart_e_ext * (NCOL + 2)];
    double *x_ps_temp = new double[Npart_p_ext * (NCOL + 2)];

    double *x_es_temp_total = new double[Npart_e_ext * (NCOL + 2)];
    double *x_ps_temp_total = new double[Npart_p_ext * (NCOL + 2)];

if(ebunch_index != -1)
    cudaMemcpy(x_es_temp, dx_es, sizeof(double) * Npart_e_ext * (NCOL + 2), cudaMemcpyDeviceToHost);

if(pbunch_index != -1)
    cudaMemcpy(x_ps_temp, dx_ps, sizeof(double) * Npart_p_ext * (NCOL + 2), cudaMemcpyDeviceToHost);

    double x,y;
    int slice_id;

if(ebunch_index != -1){
    for(int i = 0; i < Npart_e_ext; i++)
    {
        slice_id = x_es_temp[(6 * Npart_e_ext) + i];

        if(slice_id != -1)
        {
            x = x_es_temp[i];
            y = x_es_temp[(2 * Npart_e_ext) + i];

            x_es_temp[i] = (x - xbar_e[slice_id]) * (x - xbar_e[slice_id]);
            x_es_temp[(2 * Npart_e_ext) + i] = (y - ybar_e[slice_id]) * (y - ybar_e[slice_id]);

            x_es_temp_total[i] = (x - xbar_e_total) * (x - xbar_e_total);
            x_es_temp_total[(2 * Npart_e_ext) + i] = (y - ybar_e_total) * (y - ybar_e_total);
        }
    }
}

if(pbunch_index != -1){
    for(int i = 0; i < Npart_p_ext; i++)
    {
        slice_id = x_ps_temp[(6 * Npart_p_ext) + i];

        if(slice_id != -1)
        {
            x = x_ps_temp[i];
            y = x_ps_temp[(2 * Npart_p_ext) + i];

            x_ps_temp[i] = (x - xbar_p[slice_id]) * (x - xbar_p[slice_id]);
            x_ps_temp[(2 * Npart_p_ext) + i] = (y - ybar_p[slice_id]) * (y - ybar_p[slice_id]);

            x_ps_temp_total[i] = (x - xbar_p_total) * (x - xbar_p_total);
            x_ps_temp_total[(2 * Npart_p_ext) + i] = (y - ybar_p_total) * (y - ybar_p_total);
        }
    }
}

    double *dx_es_temp, *dx_ps_temp;

    cudaMalloc((void **)&dx_es_temp, sizeof(double) * Npart_e_ext * (NCOL + 2));
    cudaMalloc((void **)&dx_ps_temp, sizeof(double) * Npart_p_ext * (NCOL + 2));

if(ebunch_index != -1)
    cudaMemcpy(dx_es_temp, x_es_temp, sizeof(double) * Npart_e_ext * (NCOL + 2), cudaMemcpyHostToDevice);
if(pbunch_index != -1)
    cudaMemcpy(dx_ps_temp, x_ps_temp, sizeof(double) * Npart_p_ext * (NCOL + 2), cudaMemcpyHostToDevice);

if(ebunch_index != -1){
    ptr = thrust::device_pointer_cast(dx_es_temp);

    for(int i = 0; i < N; i++)
    {
        if(iSe[i] > 1)
        {
            for(int j = i - 1; j >= 0; j--)
            {
                base_index_ms_e = base_index_ms_e + iSe_ext[j];
            }


            sig_x_e[i] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + iSe[i], (double) 0, thrust::plus<double>());
            sig_y_e[i] = thrust::reduce(base_index_ms_e + ptr + 2 * Npart_e_ext, base_index_ms_e + ptr + 2 * Npart_e_ext + iSe[i], (double) 0, thrust::plus<double>());


            sig_x_e[i] = sqrt(sig_x_e[i]/double(iSe[i]));
            sig_y_e[i] = sqrt(sig_y_e[i]/double(iSe[i]));

            base_index_ms_e = 0;

        }
        else
        {
            sig_x_e[i] = 0;
            sig_y_e[i] = 0;
        }

    }
}

if(pbunch_index != -1){
    ptr = thrust::device_pointer_cast(dx_ps_temp);

    for(int i = 0; i < N; i++)
    {
        if(iSp[i] > 1)
        {
            for(int j = i - 1; j >= 0; j--)
            {
                    base_index_ms_p = base_index_ms_p + iSp_ext[j];
            }


            sig_x_p[i] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + iSp[i], (double) 0, thrust::plus<double>());
            sig_y_p[i] = thrust::reduce(base_index_ms_p +  2 * Npart_p_ext + ptr, base_index_ms_p +  2 * Npart_p_ext + ptr + iSp[i], (double) 0, thrust::plus<double>());

            sig_x_p[i] = sqrt(sig_x_p[i]/double(iSp[i]));
            sig_y_p[i] = sqrt(sig_y_p[i]/double(iSp[i]));

            base_index_ms_p = 0;

        }
        else
        {
            sig_x_p[i] = 0;
            sig_y_p[i] = 0;
        }
    }
}

if(ebunch_index != -1)
    cudaMemcpy(dx_es_temp, x_es_temp_total, sizeof(double) * Npart_e_ext * (NCOL + 2), cudaMemcpyHostToDevice);
if(pbunch_index != -1)
    cudaMemcpy(dx_ps_temp, x_ps_temp_total, sizeof(double) * Npart_p_ext * (NCOL + 2), cudaMemcpyHostToDevice);

int iSe_total = 0, iSp_total = 0;

if(ebunch_index != -1){
    ptr = thrust::device_pointer_cast(dx_es_temp);

    for(int i = 0; i < N; i++)
    {
        if(iSe[i] > 1)
        {
            for(int j = i - 1; j >= 0; j--)
            {
                base_index_ms_e = base_index_ms_e + iSe_ext[j];
            }


            sig_x_e_total+= thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + iSe[i], (double) 0, thrust::plus<double>());
            sig_y_e_total+= thrust::reduce(base_index_ms_e + ptr + 2 * Npart_e_ext, base_index_ms_e + ptr + 2 * Npart_e_ext + iSe[i], (double) 0, thrust::plus<double>());

            base_index_ms_e = 0;
        }
        iSe_total += iSe[i];
    }

    sig_x_e_total = sqrt(sig_x_e_total/double(iSe_total));
    sig_y_e_total = sqrt(sig_y_e_total/double(iSe_total));
}

if(pbunch_index != -1){
    ptr = thrust::device_pointer_cast(dx_ps_temp);

    for(int i = 0; i < N; i++)
    {
        if(iSp[i] > 1)
        {
            for(int j = i - 1; j >= 0; j--)
            {
                    base_index_ms_p = base_index_ms_p + iSp_ext[j];
            }


            sig_x_p_total += thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + iSp[i], (double) 0, thrust::plus<double>());
            sig_y_p_total += thrust::reduce(base_index_ms_p +  2 * Npart_p_ext + ptr, base_index_ms_p +  2 * Npart_p_ext + ptr + iSp[i], (double) 0, thrust::plus<double>());

            base_index_ms_p = 0;

        }
        iSp_total += iSp[i];
    }

    sig_x_p_total = sqrt(sig_x_p_total/double(iSp_total));
    sig_y_p_total = sqrt(sig_y_p_total/double(iSp_total));
}
    
    std::string filename_e, filename_p;

    std::stringstream e_s, p_s;
	
    e_s << ebunch_index;
    p_s << pbunch_index;

    filename_e = "output/beamspecs_e_" + e_s.str() + ".out";
    filename_p = "output/beamspecs_p_" + p_s.str() + ".out";


    std::stringstream ss_e, ss_p;
    ss_e.precision(12);
    ss_p.precision(12);
    //std::cout<<"Sig_x_e[i]\n";
    //std::cout<<sig_x_e[0]<<std::endl;
    //std::cout<<sig_x_e[1]<<std::endl;
    //std::cout<<sig_x_e[2]<<std::endl;
    //std::cout<<"Sig_x_e_total:"<<sig_x_e_total<<std::endl;
if(ebunch_index != -1)
    ss_e << lum_turn << "\t" << xbar_e_total << "\t" << ybar_e_total << "\t" << sig_x_e_total << "\t" << sig_y_e_total << "\t";
if(pbunch_index != -1)
    ss_p << lum_turn << "\t" << xbar_p_total << "\t" << ybar_p_total << "\t" << sig_x_p_total << "\t" << sig_y_p_total << "\t";

    for(int i = 0; i < N; i++)
    {
if(ebunch_index != -1)
        ss_e << xbar_e[i] << "\t" << ybar_e[i] << "\t" << sig_x_e[i] << "\t" << sig_y_e[i] << "\t";
if(pbunch_index != -1)
        ss_p << xbar_p[i] << "\t" << ybar_p[i] << "\t" << sig_x_p[i] << "\t" << sig_y_p[i] << "\t";
    }

if(ebunch_index != -1)
    ss_e << "\n";
if(pbunch_index != -1)
    ss_p << "\n";


    std::_Ios_Openmode mode = std::ios::app;

if(ebunch_index != -1){
    std::ofstream file(filename_e.c_str(), mode);
    if (!file.is_open()){
        std::string msg = "Cannot open file " + filename_e;
    }

    file << ss_e.str();
    file.close();
}

if(pbunch_index != -1){
    std::ofstream file1(filename_p.c_str(), mode);
    if (!file1.is_open()){
        std::string msg = "Cannot open file " + filename_p;
    }

    file1 << ss_p.str();
    file1.close();
}

    cudaFree(dx_es_temp);
    cudaFree(dx_ps_temp);

    delete[] x_es_temp;
    delete[] x_ps_temp;
    delete[] xbar_e;
    delete[] ybar_e;
    delete[] xbar_p;
    delete[] ybar_p;
    delete[] sig_x_e;
    delete[] sig_y_e;
    delete[] sig_x_p;
    delete[] sig_y_p;
    delete[] x_es_temp_total;
    delete[] x_ps_temp_total;
}

void Collide::collide_back(int *dOutOfBound_e, int *dOutOfBound_p, double *dx_e, double *dx_p, BeamParams *bParams, int e_bunch, int opp_p_bunch, int opp_p_bunch_gpu, int p_bunch, int opp_e_bunch, int opp_e_bunch_gpu){
        //std::cout<<"Came here start"<<" rank is "<<world_rank<<std::endl;
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	return;
}

void dumpSlices(int N, int Npart_e, int Npart_p, int *iSize_e, int *iSize_p, BeamParams *bParams, double *xbar_e_dump, double *ybar_e_dump, double *sig_x_e_dump, double *sig_y_e_dump,  double *xbar_p_dump, double *ybar_p_dump, double *sig_x_p_dump,  double *sig_y_p_dump)
{
  double Lsl = 0.0;

  int row_count = 0;
  double sig_x, sig_y, arg;
  //cout<<"calculating lsl1"<<endl;
  //cout<<"isp:"<<iSize_p[0]<<endl;
  for(int i = 2; i <= N+1; ++i){
    for(int j = 1; j <= i-1; ++j)
	{	 
        sig_x = sqrt(pow(sig_x_e_dump[(row_count*N) + j-1],2) + pow(sig_x_p_dump[(row_count*N) + i-j-1],2));
        sig_y = sqrt(pow(sig_y_e_dump[(row_count*N) + j-1],2) + pow(sig_y_p_dump[(row_count*N) + i-j-1],2));
		
        arg = exp(-0.5*(pow(((xbar_p_dump[(row_count*N) + i-j-1] -xbar_e_dump[(row_count*N) + j-1])/sig_x),2)) - 0.5*(pow(((ybar_p_dump[(row_count*N) + i-j-1] -ybar_e_dump[(row_count*N) + j-1])/sig_y),2)));
		
        if((iSize_e[j-1] > 0) && (iSize_p[i-j-1] > 0))
            Lsl = Lsl + arg*iSize_e[j-1]*iSize_p[i-j-1]/(sig_x*sig_y);
			
    }
    row_count++;
	
  }
  
  //cout<<"finished 1st for loop\n";
  for(int i = N+2; i <= 2*N; ++i){
    for(int j = i-N; j <= N; ++j){
        sig_x = sqrt(pow(sig_x_e_dump[(row_count*N) + j-1],2) + pow(sig_x_p_dump[(row_count*N) + i-j-1],2));
        sig_y = sqrt(pow(sig_y_e_dump[(row_count*N) + j-1],2) + pow(sig_y_p_dump[(row_count*N) + i-j-1],2));

        arg = exp(-0.5*(pow(((xbar_p_dump[(row_count*N) + i-j-1] -xbar_e_dump[(row_count*N) + j-1])/sig_x),2)) - 0.5*(pow(((ybar_p_dump[(row_count*N) + i-j-1] -ybar_e_dump[(row_count*N) + j-1])/sig_y),2)));

        if((iSize_e[j-1] > 0) && (iSize_p[i-j-1] > 0)){
                Lsl = Lsl + arg*iSize_e[j-1]*iSize_p[i-j-1]/(sig_x*sig_y);
        }
    }
    row_count++;
	
  }
  
  bParams->Lsl = bParams->Lc*Lsl/(double(Npart_e)*double(Npart_p));
 
}

void Collide::collide(int lum_turn, int *dOutOfBound_e, int *dOutOfBound_p, double *dx_e, double *dx_p, BeamParams *bParams, int e_bunch, int opp_p_bunch, int opp_p_bunch_gpu, int p_bunch, int opp_e_bunch, int opp_e_bunch_gpu){

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	//cout<<"Inside Collide:"<<e_bunch<<"-["<<p_bunch<<"]"<<endl;

	double zmin_e, L_e, delta_e, *z_e, *z_p_rcvd, *SS_e;
	double zmin_p, L_p, delta_p, *z_p, *z_e_rcvd, *SS_p;
	
	int N = bParams->N;
	
	double *dS_e, *dS_p;
	
	if(e_bunch != -1){
		z_e = new double[N];
		z_p_rcvd = new double[N];
		SS_e = new double[N*N];
	
		double sig_z_e = find_zmin(dx_e, bParams->Npart_e);	
		L_e = 6.0 * sig_z_e;
		zmin_e = -0.5*L_e;
		
		if(N == 1){
			delta_e = 0.0;
			z_e[0] = 0.0;
		}
		else{
			delta_e = L_e/(double)N;
			int je = 0;
			for(int i = -(N - 1); i <= (N - 1); i+=2){
				je = je + 1;
				z_e[je - 1] = -i * delta_e/2.0;
			}
		}
	}
	
	if(p_bunch != -1){
		z_p = new double[N];
		z_e_rcvd = new double[N];
		SS_p = new double[N*N];
	
		double sig_z_p = find_zmin(dx_p, bParams->Npart_p);	
		L_p = 6.0 * sig_z_p;
		zmin_p = -0.5*L_p;
		
		if(N == 1){
			delta_p = 0.0;
			z_p[0] = 0.0;
		}
		else{
			delta_p = L_p/(double)N;
			int jp = 0;
			for(int i = -(N - 1); i <= (N - 1); i+=2){
				jp = jp + 1;
				z_p[jp - 1] = -i * delta_p/2.0;
			}
		}
	}

	//std::cout<<"Came here start 111 \n";

	//MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Request myRequest_send_e[6], myRequest_recv_e[6], myRequest_send_p[6], myRequest_recv_p[6];
	MPI_Status status_send_e[6], status_recv_e[6], status_send_p[6], status_recv_p[6];

  	//quad::timer::event_pair timer_comm1;
  	//quad::timer::start_timer(&timer_comm1);

	//std::cout<<"Came here start 111-111 \n";
	
	if(e_bunch != -1){

		if(opp_p_bunch_gpu != world_rank){
			MPI_Isend(z_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_send_e[0]);
		
			MPI_Irecv(z_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_recv_e[0]);
		}
		else{
			memcpy(z_p_rcvd, z_p, sizeof(double) * N);
		}
		//send z_e to opp_p_bunch_gpu
		//receieve z_p from opp_p_bunch_gpu
		//store z_p in z_p_recvd
		//calculate SS_e
	}
	
	if(p_bunch != -1){

		if(opp_e_bunch_gpu != world_rank){
			MPI_Isend(z_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_send_p[0]);
		
			MPI_Irecv(z_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_recv_p[0]);
		}
		else{
			memcpy(z_e_rcvd, z_e, sizeof(double) * N);
		}
		//send z_p to opp_e_bunch_gpu
		//receieve z_e from opp_e_bunch_gpu
		//store z_e in z_e_recvd
		//calculate SS_p
	}
	
	if(e_bunch != -1){
		if(opp_p_bunch_gpu != world_rank){
			MPI_Wait(&myRequest_send_e[0], &status_send_e[0]);
			MPI_Wait(&myRequest_recv_e[0], &status_recv_e[0]);
		}
	}
	
	if(p_bunch != -1){
		if(opp_e_bunch_gpu != world_rank){
			MPI_Wait(&myRequest_send_p[0], &status_send_p[0]);
			MPI_Wait(&myRequest_recv_p[0], &status_recv_p[0]);
		}
	}

	//quad::timer::stop_timer(&timer_comm1, "Comm1");

	//std::cout<<"Came here start 222 \n";

	//MPI_Barrier(MPI_COMM_WORLD);
	
	int k = 0;
	if(e_bunch != -1){
		for(int i = 2; i <= N+1; ++i){
			for(int j = 1; j <= i-1; ++j){
				SS_e[k] = (z_p_rcvd[i-j-1]-z_e[j-1])/2.0;
				k++;
			}
		}
		
		for(int i = N+2; i <= 2*N; ++i){
			for(int j = i - N; j<= N; ++j){
				SS_e[k] = (z_p_rcvd[i-j-1]-z_e[j-1])/2.0;
				k++;
			}
		}
		
		cudaMalloc((void **)&dS_e, sizeof(double) * N * N);
		cudaMemcpy(dS_e, SS_e, sizeof(double) * N * N, cudaMemcpyHostToDevice);
	}
	
	k = 0;
	if(p_bunch != -1){
		for(int i = 2; i <= N+1; ++i){
			for(int j = 1; j <= i-1; ++j){
				SS_p[k] = (z_p[i-j-1]-z_e_rcvd[j-1])/2.0;
				k++;
			}
		}
		
		for(int i = N+2; i <= 2*N; ++i){
			for(int j = i - N; j<= N; ++j){
				SS_p[k] = (z_p[i-j-1]-z_e_rcvd[j-1])/2.0;
				k++;
			}
		}
		
		cudaMalloc((void **)&dS_p, sizeof(double) * N * N);
		cudaMemcpy(dS_p, SS_p, sizeof(double) * N * N, cudaMemcpyHostToDevice);
	}

	//std::cout<<"Came here start 333 \n";

        //MPI_Barrier(MPI_COMM_WORLD);

	//std::cout<<"After Barrier \n";

	
	int *d_iSe, *d_iSp, *d_iSe_ext, *d_iSp_ext, *iSe, *iSp, *iSe_ext, *iSp_ext;
	int Npart_e_ext = 0, Npart_p_ext = 0;
	
	int *d_iSe_Inc, *d_iSp_Inc;
	double *dx_es, *dx_ps, *dx_ems, *dx_pms, *dx_esd, *dx_psd;
	
	double *xbar_e, *ybar_e, *xbar_p, *ybar_p, *xpbar_e, *ypbar_e, *xpbar_p, *ypbar_p;
	double *sig_x_e, *sig_y_e, *sig_x_p, *sig_y_p;
	
	double *dxbar_e, *dybar_e, *dxbar_p, *dybar_p, *dxpbar_e, *dypbar_e, *dxpbar_p, *dypbar_p;
	double *d_sig_x_e, *d_sig_y_e, *d_sig_x_p, *d_sig_y_p;
	
	double *dx_emb, *dx_pmb, *dx_esb, *dx_psb;

	int base_index_ms_e = 0, base_index_ms_p = 0;
	int numThreads, numBlocks;
	
	thrust::device_ptr<double> ptr;
	
	if(e_bunch != -1){
		iSe = new int[N];
		iSe_ext = new int[N];
		
		cudaMalloc((void **)&d_iSe, sizeof(int) * N);
		cudaMemset(d_iSe, 0, N * sizeof(int));
		
		cudaMalloc((void **)&d_iSe_ext, sizeof(int) * N);
		
		numThreads = BLOCKDIMX;
		numBlocks = bParams->Npart_e/numThreads + ((bParams->Npart_e%numThreads)?1:0);
		cout<<"Before Slice E:"<<numBlocks<<","<<numThreads<<endl;
		slice<<<numBlocks, numThreads>>>(dx_e, zmin_e, bParams->Npart_e, bParams->N, d_iSe, dOutOfBound_e);

		//std::cout<<"Finished slicing \n";
		
		cudaMemcpy(iSe, d_iSe, sizeof(int) * N, cudaMemcpyDeviceToHost);
		
		for(int i = 0; i < N; i++){
			if(iSe[i]%BLOCKDIMX_COL == 0){
				iSe_ext[i] = iSe[i];
			}
			else{
				iSe_ext[i] = (iSe[i]/BLOCKDIMX_COL + 1) * BLOCKDIMX_COL;
			}

			Npart_e_ext += iSe_ext[i];
		}
		cudaMemcpy(d_iSe_ext, iSe_ext, sizeof(int) * N, cudaMemcpyHostToDevice);
		
		cudaMalloc((void **)&d_iSe_Inc, sizeof(int) * N);
		cudaMemset(d_iSe_Inc, 0, N * sizeof(int));
		
		cudaMalloc((void **)&dx_es, sizeof(double) * Npart_e_ext * (NCOL + 2));
		
		ptr = thrust::device_pointer_cast(dx_es);
		thrust::fill(ptr, ptr + Npart_e_ext * (NCOL + 2), -1);
		
		cudaMalloc((void **)&dx_ems, sizeof(double) * Npart_e_ext * 3);
		cudaMalloc((void **)&dx_esd, sizeof(double) * Npart_e_ext * 3);
		cout<<"Before Sort E:"<<numBlocks<<","<<numThreads<<endl;

		sortbeam<<<numBlocks, numThreads>>>(dx_e, dx_es, d_iSe, d_iSe_Inc, d_iSe_ext, bParams->Npart_e, Npart_e_ext, NCOL, dS_e, dx_ems, dx_esd, N, 1);

		//std::cout<<"Finished Sorting \n";
		
		xbar_e = new double[N];
		ybar_e = new double[N];
		sig_x_e = new double[N];
		sig_y_e = new double[N];

  		xpbar_e = new double[N];
  		ypbar_e = new double[N];
		
		ptr = thrust::device_pointer_cast(dx_ems);
		
		for(int i = 0; i < N; i++){
			if(iSe[i] > 1){
				for(int j = i - 1; j >= 0; j--){
					base_index_ms_e = base_index_ms_e + iSe_ext[j];
				}

				xbar_e[i] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + iSe[i], (double) 0, thrust::plus<double>())/iSe[i];
				ybar_e[i] = thrust::reduce(base_index_ms_e + ptr + Npart_e_ext, base_index_ms_e + ptr + Npart_e_ext + iSe[i], (double) 0, thrust::plus<double>())/iSe[i];

				base_index_ms_e = 0;
			}
			else{
				xbar_e[i] = 0;
				ybar_e[i] = 0;
			}
		}
		
		ptr = thrust::device_pointer_cast(dx_esd);

		for(int i = 0; i < N; i++){
			if(iSe[i] > 1){
                for(int j = i - 1; j >= 0; j--){
					base_index_ms_e = base_index_ms_e + iSe_ext[j];
				}
				
				sig_x_e[i] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + iSe[i], (double) 0, thrust::plus<double>());
				sig_y_e[i] = thrust::reduce(base_index_ms_e + ptr + Npart_e_ext, base_index_ms_e + ptr + Npart_e_ext + iSe[i], (double) 0, thrust::plus<double>());

				sig_x_e[i] = sqrt((sig_x_e[i] - (iSe[i] * xbar_e[i] * xbar_e[i]))/iSe[i]);
				sig_y_e[i] = sqrt((sig_y_e[i] - (iSe[i] * ybar_e[i] * ybar_e[i]))/iSe[i]);

				base_index_ms_e = 0;
			}
			else{
				sig_x_e[i] = 0;
				sig_y_e[i] = 0;
			}
		}

  		ptr = thrust::device_pointer_cast(dx_es);

  		for(int i = 0; i < N; i++)
  		{
        		if(iSe[i] > 1)
        		{
                		for(int j = i - 1; j >= 0; j--)
                		{
                        		base_index_ms_e = base_index_ms_e + iSe_ext[j];
                		}


                		xpbar_e[i] = thrust::reduce(base_index_ms_e + ptr + Npart_e_ext, base_index_ms_e + ptr + Npart_e_ext + iSe[i], (double) 0, thrust::plus<double>())/iSe[i];
                		ypbar_e[i] = thrust::reduce(base_index_ms_e + ptr + (3 * Npart_e_ext), base_index_ms_e + ptr + (3 * Npart_e_ext) + iSe[i], (double) 0, thrust::plus<double>())/iSe[i];

                		base_index_ms_e = 0;
        		}
        		else
        		{
                		xpbar_e[i] = 0;
                		ypbar_e[i] = 0;
        		}

  		}
		
		cudaMalloc((void **)&dx_emb, sizeof(double) * (Npart_e_ext/BLOCKDIMX_COL) * 2);
		cudaMemset(dx_emb, 0, sizeof(double) * (Npart_e_ext/BLOCKDIMX_COL) * 2);
		
		cudaMalloc((void **)&dx_esb, sizeof(double) * (Npart_e_ext/BLOCKDIMX_COL) * 2);
		cudaMemset(dx_esb, 0, sizeof(double) * (Npart_e_ext/BLOCKDIMX_COL) * 2);

		//std::cout<<"Finished Mean and SD \n";
	}
	
	if(p_bunch != -1){
		iSp = new int[N];
		iSp_ext = new int[N];
		
		cudaMalloc((void **)&d_iSp, sizeof(int) * N);
		cudaMemset(d_iSp, 0, N * sizeof(int));
		
		cudaMalloc((void **)&d_iSp_ext, sizeof(int) * N);
		
		numThreads = BLOCKDIMX;
		numBlocks = bParams->Npart_p/numThreads + ((bParams->Npart_p%numThreads)?1:0);
		cout<<"Before Slice P:"<<numBlocks<<","<<numThreads<<endl;

		slice<<<numBlocks, numThreads>>>(dx_p, zmin_p, bParams->Npart_p, bParams->N, d_iSp, dOutOfBound_p);

		//std::cout<<"Finished slicing of p-beam \n";
		
		cudaMemcpy(iSp, d_iSp, sizeof(int) * N, cudaMemcpyDeviceToHost);
		
		for(int i = 0; i < N; i++){
			if(iSp[i]%BLOCKDIMX_COL == 0){
				iSp_ext[i] = iSp[i];
			}
			else{
				iSp_ext[i] = (iSp[i]/BLOCKDIMX_COL + 1) * BLOCKDIMX_COL;
			}

			Npart_p_ext += iSp_ext[i];
		}
		cudaMemcpy(d_iSp_ext, iSp_ext, sizeof(int) * N, cudaMemcpyHostToDevice);
		
		cudaMalloc((void **)&d_iSp_Inc, sizeof(int) * N);
		cudaMemset(d_iSp_Inc, 0, N * sizeof(int));
		
		cudaMalloc((void **)&dx_ps, sizeof(double) * Npart_p_ext * (NCOL + 2));
		
		ptr = thrust::device_pointer_cast(dx_ps);
		thrust::fill(ptr, ptr + Npart_p_ext * (NCOL + 2), -1);
		
		cudaMalloc((void **)&dx_pms, sizeof(double) * Npart_p_ext * 3);
		cudaMalloc((void **)&dx_psd, sizeof(double) * Npart_p_ext * 3);

		//std::cout<<"Before sorting of p-beam \n";
		cout<<"Before sort P:"<<numBlocks<<","<<numThreads<<endl;

		sortbeam<<<numBlocks, numThreads>>>(dx_p, dx_ps, d_iSp, d_iSp_Inc, d_iSp_ext, bParams->Npart_p, Npart_p_ext, NCOL, dS_p, dx_pms, dx_psd, N, 0);

		//std::cout<<"Finished sorting of p-beam \n";
		
		xbar_p = new double[N];
		ybar_p = new double[N];
		sig_x_p = new double[N];
		sig_y_p = new double[N];

                xpbar_p = new double[N];
                ypbar_p = new double[N];
		
		ptr = thrust::device_pointer_cast(dx_pms);

		for(int i = 0; i < N; i++){
			if(iSp[i] > 1){
				for(int j = i - 1; j >= 0; j--){
					base_index_ms_p = base_index_ms_p + iSp_ext[j];
				}

				xbar_p[i] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + iSp[i], (double) 0, thrust::plus<double>())/iSp[i];
				ybar_p[i] = thrust::reduce(base_index_ms_p + Npart_p_ext + ptr, base_index_ms_p + Npart_p_ext + ptr + iSp[i], (double) 0, thrust::plus<double>())/iSp[i];

				base_index_ms_p = 0;
			}
			else{
				xbar_p[i] = 0;
				ybar_p[i] = 0;
			}
		}
		
		ptr = thrust::device_pointer_cast(dx_psd);

		for(int i = 0; i < N; i++){
			if(iSp[i] > 1){
				for(int j = i - 1; j >= 0; j--){
					base_index_ms_p = base_index_ms_p + iSp_ext[j];
				}

                sig_x_p[i] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + iSp[i], (double) 0, thrust::plus<double>());
                sig_y_p[i] = thrust::reduce(base_index_ms_p + Npart_p_ext + ptr, base_index_ms_p + Npart_p_ext + ptr + iSp[i], (double) 0, thrust::plus<double>());

                sig_x_p[i] = sqrt((sig_x_p[i] - (iSp[i] * xbar_p[i] * xbar_p[i]))/iSp[i]);
                sig_y_p[i] = sqrt((sig_y_p[i] - (iSp[i] * ybar_p[i] * ybar_p[i]))/iSp[i]);

                base_index_ms_p = 0;
			}
			else{
				sig_x_p[i] = 0;
				sig_y_p[i] = 0;
			}
		}

  		ptr = thrust::device_pointer_cast(dx_ps);

  		for(int i = 0; i < N; i++)
  		{
        		if(iSp[i] > 1)
        		{
               	 		for(int j = i - 1; j >= 0; j--)
                		{
                        		base_index_ms_p = base_index_ms_p + iSp_ext[j];
                		}


                		xpbar_p[i] = thrust::reduce(base_index_ms_p + ptr + Npart_p_ext, base_index_ms_p + ptr + Npart_p_ext + iSp[i], (double) 0, thrust::plus<double>())/iSp[i];
                		ypbar_p[i] = thrust::reduce(base_index_ms_p + (3 * Npart_p_ext) + ptr, base_index_ms_p + (3 * Npart_p_ext) + ptr + iSp[i], (double) 0, thrust::plus<double>())/iSp[i];

                		base_index_ms_p = 0;
        		}
        		else
        		{
                		xpbar_p[i] = 0;
                		ypbar_p[i] = 0;
        		}
  		}

		
		cudaMalloc((void **)&dx_pmb, sizeof(double) * (Npart_p_ext/BLOCKDIMX_COL) * 2);
		cudaMemset(dx_pmb, 0, sizeof(double) * (Npart_p_ext/BLOCKDIMX_COL) * 2);
		
		cudaMalloc((void **)&dx_psb, sizeof(double) * (Npart_p_ext/BLOCKDIMX_COL) * 2);
		cudaMemset(dx_psb, 0, sizeof(double) * (Npart_p_ext/BLOCKDIMX_COL) * 2);

		//std::cout<<"Finished Mean and SD of p-beam \n";
	}

/*
	if(e_bunch != -1){
        for(int k = 0; k < N; k++) {
                std::cout<<"k is : "<<k<<"updated xbar_e is : "<<xbar_e[k]<<"\n";
                std::cout<<"k is : "<<k<<"updated ybar_e is : "<<ybar_e[k]<<"\n";
                std::cout<<"k is : "<<k<<"updated sig_x_e is : "<<sig_x_e[k]<<"\n";
                std::cout<<"k is : "<<k<<"updated sig_y_e is : "<<sig_y_e[k]<<"\n";
        }}
*/
	//std::cout<<"Came here 11 \n";

	double *xbar_e_rcvd, *ybar_e_rcvd, *xbar_p_rcvd, *ybar_p_rcvd;
	double *sig_x_e_rcvd, *sig_y_e_rcvd, *sig_x_p_rcvd, *sig_y_p_rcvd;
	int *iSe_rcvd, *iSp_rcvd;
	int *d_iSe_rcvd, *d_iSp_rcvd;
	int Npart_inbound_e_rcvd, Npart_inbound_p_rcvd;

        //quad::timer::event_pair timer_comm2;
        //quad::timer::start_timer(&timer_comm2);
	
	if(e_bunch != -1){
	
		xbar_p_rcvd = new double[N];
		ybar_p_rcvd = new double[N];
		sig_x_p_rcvd = new double[N];
		sig_y_p_rcvd = new double[N];
		iSp_rcvd = new int[N];
		
		//store receieved in xbar_p_rcvd, sig_x_p_rcvd
		
		if(opp_p_bunch_gpu != world_rank){
			//send e-beam mean and sd to opp_p_bunch GPU
			//receieve p-beam mean and sd from opp_p_bunch GPU
			//receieve iSp from opp_p_bunch GPU
			//receieve Npart_ibound_p from opp_p_bunch GPU

			int Npart_inbound_e = bParams->Npart_inbound_e;
			
			MPI_Isend(xbar_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_send_e[0]);
			MPI_Isend(ybar_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_send_e[1]);
			MPI_Isend(sig_x_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 3, MPI_COMM_WORLD, &myRequest_send_e[2]);
			MPI_Isend(sig_y_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_send_e[3]);
			MPI_Isend(iSe, N, MPI_INT, opp_p_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_send_e[4]);
			MPI_Isend(&Npart_inbound_e, 1, MPI_INT, opp_p_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_send_e[5]);
			
			MPI_Irecv(xbar_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_recv_e[0]);
			MPI_Irecv(ybar_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_recv_e[1]);
			MPI_Irecv(sig_x_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 9, MPI_COMM_WORLD, &myRequest_recv_e[2]);
			MPI_Irecv(sig_y_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 10, MPI_COMM_WORLD, &myRequest_recv_e[3]);
			MPI_Irecv(iSp_rcvd, N, MPI_INT, opp_p_bunch_gpu, 11, MPI_COMM_WORLD, &myRequest_recv_e[4]);
			MPI_Irecv(&Npart_inbound_p_rcvd, 1, MPI_INT, opp_p_bunch_gpu, 12, MPI_COMM_WORLD, &myRequest_recv_e[5]);
		}
		else{
			memcpy(xbar_p_rcvd, xbar_p, sizeof(double) * N);
			memcpy(ybar_p_rcvd, ybar_p, sizeof(double) * N);
			memcpy(sig_x_p_rcvd, sig_x_p, sizeof(double) * N);
			memcpy(sig_y_p_rcvd, sig_y_p, sizeof(double) * N);
			
			memcpy(iSp_rcvd, iSp, sizeof(int) * N);
			Npart_inbound_p_rcvd = bParams->Npart_inbound_p;
		}
	}
	//cout<<"stuff"<<endl;
	if(p_bunch != -1){
		
		xbar_e_rcvd = new double[N];
		ybar_e_rcvd = new double[N];
		sig_x_e_rcvd = new double[N];
		sig_y_e_rcvd = new double[N];
		iSe_rcvd = new int[N];
		
		if(opp_e_bunch_gpu != world_rank){
			//send p-beam mean and sd to opp_e_bunch GPU
			//receieve e-beam mean and sd from opp_e_bunch GPU
			//store receieved in xbar_e_rcvd, sig_x_e_rcvd
			//receieve iSe from opp_e_bunch GPU
			//receieve Npart_ibound_e from opp_e_bunch GPU

			int Npart_inbound_p = bParams->Npart_inbound_p;
			
			MPI_Isend(xbar_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_send_p[0]);
			MPI_Isend(ybar_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_send_p[1]);
			MPI_Isend(sig_x_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 9, MPI_COMM_WORLD, &myRequest_send_p[2]);
			MPI_Isend(sig_y_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 10, MPI_COMM_WORLD, &myRequest_send_p[3]);
			MPI_Isend(iSp, N, MPI_INT, opp_e_bunch_gpu, 11, MPI_COMM_WORLD, &myRequest_send_p[4]);
			MPI_Isend(&Npart_inbound_p, 1, MPI_INT, opp_e_bunch_gpu, 12, MPI_COMM_WORLD, &myRequest_send_p[5]);
			
			MPI_Irecv(xbar_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_recv_p[0]);
			MPI_Irecv(ybar_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_recv_p[1]);
			MPI_Irecv(sig_x_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 3, MPI_COMM_WORLD, &myRequest_recv_p[2]);
			MPI_Irecv(sig_y_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_recv_p[3]);
			MPI_Irecv(iSe_rcvd, N, MPI_INT, opp_e_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_recv_p[4]);
			MPI_Irecv(&Npart_inbound_e_rcvd, 1, MPI_INT, opp_e_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_recv_p[5]);
		}
		else{
			memcpy(xbar_e_rcvd, xbar_e, sizeof(double) * N);
			memcpy(ybar_e_rcvd, ybar_e, sizeof(double) * N);
			memcpy(sig_x_e_rcvd, sig_x_e, sizeof(double) * N);
			memcpy(sig_y_e_rcvd, sig_y_e, sizeof(double) * N);
			
			memcpy(iSe_rcvd, iSe, sizeof(int) * N);
			Npart_inbound_e_rcvd = bParams->Npart_inbound_e;
		}
	}
	
	//DO MPI_Wait calls here
	
	if(e_bunch != -1){
		if(opp_p_bunch_gpu != world_rank){
			MPI_Waitall(6, myRequest_send_e, status_send_e);
			MPI_Waitall(6, myRequest_recv_e, status_recv_e);
		}
	}
	
	if(p_bunch != -1){
		if(opp_e_bunch_gpu != world_rank){
			MPI_Waitall(6, myRequest_send_p, status_send_p);
			MPI_Waitall(6, myRequest_recv_p, status_recv_p);
		}
	}
        //cout<<"performed comm2\n";
	//quad::timer::stop_timer(&timer_comm2, "Comm2");
        //cudaError_t errorA = cudaGetLastError();
        //if(errorA != cudaSuccess)
          {
            // something's gone wrong
            // print out the CUDA error as a string
            // printf("CUDA ErrorA: %s\n bunches:%i %i", cudaGetErrorString(errorA), e_bunch, p_bunch);

            // we can't recover from the error -- exit the program
           
          }
	if(e_bunch != -1){
		(cudaMalloc((void **)&dxbar_p, sizeof(double) * N));
		(cudaMalloc((void **)&dybar_p, sizeof(double) * N));

  		(cudaMalloc((void **)&dxpbar_e, sizeof(double) * N));
  		(cudaMalloc((void **)&dypbar_e, sizeof(double) * N));
		
		(cudaMalloc((void **)&d_sig_x_p, sizeof(double) * N));
		(cudaMalloc((void **)&d_sig_y_p, sizeof(double) * N));
		
		(cudaMalloc((void **)&d_iSp_rcvd, sizeof(int) * N));
	
		(cudaMemcpy(dxbar_p, xbar_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		(cudaMemcpy(dybar_p, ybar_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
                
  		(cudaMemcpy(dxpbar_e, xpbar_e, sizeof(double) * N, cudaMemcpyHostToDevice));
  		(cudaMemcpy(dypbar_e, ypbar_e, sizeof(double) * N, cudaMemcpyHostToDevice));
		(cudaMemcpy(d_sig_x_p, sig_x_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		(cudaMemcpy(d_sig_y_p, sig_y_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		(cudaMemcpy(d_iSp_rcvd, iSp_rcvd, sizeof(int) * N, cudaMemcpyHostToDevice));
        
	}
	
	if(p_bunch != -1){
          
		(cudaMalloc((void **)&dxbar_e, sizeof(double) * N));
		(cudaMalloc((void **)&dybar_e, sizeof(double) * N));

  		(cudaMalloc((void **)&dxpbar_p, sizeof(double) * N));
  		(cudaMalloc((void **)&dypbar_p, sizeof(double) * N));

		(cudaMalloc((void **)&d_sig_x_e, sizeof(double) * N));
		(cudaMalloc((void **)&d_sig_y_e, sizeof(double) * N));
		
		(cudaMalloc((void **)&d_iSe_rcvd, sizeof(int) * N));
                
		(cudaMemcpy(dxbar_e, xbar_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		(cudaMemcpy(dybar_e, ybar_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
                
  		(cudaMemcpy(dxpbar_p, xpbar_p, sizeof(double) * N, cudaMemcpyHostToDevice));
  		(cudaMemcpy(dypbar_p, ypbar_p, sizeof(double) * N, cudaMemcpyHostToDevice));
                
		(cudaMemcpy(d_sig_x_e, sig_x_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		(cudaMemcpy(d_sig_y_e, sig_y_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
	     
                (cudaMemcpy(d_iSe_rcvd, iSe_rcvd, sizeof(int) * N, cudaMemcpyHostToDevice));
                
	}

	
	calculate_rms(lum_turn, e_bunch, p_bunch, dx_es, dx_ps, N, iSe, iSp, iSe_ext, iSp_ext, Npart_e_ext, Npart_p_ext, 0);
	
	int epart = 0; int ppart = 0;
	int S_Index = 0;
	
	numThreads = BLOCKDIMX_COL;
	
	double *xbar_e_dump, *ybar_e_dump, *sig_x_e_dump, *sig_y_e_dump;
	double *xbar_p_dump, *ybar_p_dump, *sig_x_p_dump, *sig_y_p_dump;

	xbar_e_dump = new double[N*(2*N-1)];
	ybar_e_dump = new double[N*(2*N-1)];
	sig_x_e_dump = new double[N*(2*N-1)];
	sig_y_e_dump = new double[N*(2*N-1)];
	xbar_p_dump = new double[N*(2*N-1)];
	ybar_p_dump = new double[N*(2*N-1)];
	sig_x_p_dump = new double[N*(2*N-1)];
	sig_y_p_dump = new double[N*(2*N-1)];
	
	int col_num_dump = 0;
	
	for(int i = 0; i < N*(2*N-1); i++)
	{
        xbar_e_dump[i] = -1;
        ybar_e_dump[i] = -1;
        sig_x_e_dump[i] = -1;
        sig_y_e_dump[i] = -1;

        xbar_p_dump[i] = -1;
        ybar_p_dump[i] = -1;
        sig_x_p_dump[i] = -1;
        sig_y_p_dump[i] = -1;
	}
	
	for(int i = 0; i < N; i++){
		
		epart = 0; ppart = 0;
		
		int k_start = 0;
		
		if(i == N - 1)
        k_start = 1;

		for(int j = 0; j < i+1; j++){
			if(e_bunch != -1)
			{
				xbar_e_dump[col_num_dump*N + j] = xbar_e[j];//added by Ioannis
				ybar_e_dump[col_num_dump*N + j] = ybar_e[j];//added by Ioannis
				sig_x_e_dump[col_num_dump*N + j] = sig_x_e[j];//added by Ioannis
				sig_y_e_dump[col_num_dump*N + j] = sig_y_e[j];//added by Ioannis
				
				if(p_bunch ==-1)
				{
					xbar_p_dump[col_num_dump*N + j] = xbar_p_rcvd[j];//added by Ioannis
					ybar_p_dump[col_num_dump*N + j] = ybar_p_rcvd[j];//added by Ioannis
					sig_x_p_dump[col_num_dump*N + j] = sig_x_p_rcvd[j];//added by Ioannis
					sig_y_p_dump[col_num_dump*N + j] = sig_y_p_rcvd[j];//added by Ioannis
					
				}
				epart += iSe_ext[j];
			}
				
			if(p_bunch != -1)
			{
				xbar_p_dump[col_num_dump*N + j] = xbar_p[j];//added by Ioannis
				ybar_p_dump[col_num_dump*N + j] = ybar_p[j];//added by Ioannis
				sig_x_p_dump[col_num_dump*N + j] = sig_x_p[j];//added by Ioannis
				sig_y_p_dump[col_num_dump*N + j] = sig_y_p[j];//added by Ioannis
				
				if(e_bunch == -1)
				{
					xbar_e_dump[col_num_dump*N + j] = xbar_e_rcvd[j];//added by Ioannis
					ybar_e_dump[col_num_dump*N + j] = ybar_e_rcvd[j];//added by Ioannis
					sig_x_e_dump[col_num_dump*N + j] = sig_x_e_rcvd[j];//added by Ioannis
					sig_y_e_dump[col_num_dump*N + j] = sig_y_e_rcvd[j];//added by Ioannis
				}
				ppart += iSp_ext[j];
			}
				
		}
		col_num_dump++;
		
		if(e_bunch != -1){
			numBlocks = (epart)/numThreads + (((epart)%numThreads)?1:0);
			
			int num_devices=-1;
			cudaGetDeviceCount(&num_devices);
			int deviceRank=-1;
			cudaDeviceProp devProp;
			cudaGetDevice(&deviceRank);
			cudaGetDeviceProperties(&devProp, deviceRank);
			//printf("Num Devices: %i\n", deviceRank);
			//printf("About to KICK MPI RANK:%i GPU Rank:%i, name:%s\n",world_rank, deviceRank, devProp.name);
			
			//printf("about to kick:%i\n", deviceRank);
                        //cout<<"Before kick E:"<<numBlocks<<","<<numThreads<<endl;

			applyKickGPU_E<<<numBlocks, numThreads>>>(dx_es, bParams->gamma_e, Npart_e_ext, dxpbar_e, dypbar_e, dxbar_p, dybar_p, d_sig_x_p, d_sig_y_p, d_iSp_rcvd, bParams->N_p, &dS_e[S_Index], i+1, epart, 0, dx_emb, dx_esb, 1, Npart_inbound_p_rcvd, bParams->N);
			
			cudaDeviceSynchronize();
			
			ptr = thrust::device_pointer_cast(dx_emb);
			
			for(int k = k_start; k <= i; k++){
				if(iSe[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_e = base_index_ms_e + (iSe_ext[j]/BLOCKDIMX_COL);
					}
					xbar_e[k] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSe[k];
					ybar_e[k] = thrust::reduce(base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL), base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL) + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSe[k];

					base_index_ms_e = 0;
				}
				else{
						xbar_e[k] = 0;
						ybar_e[k] = 0;
				}
			}
			
			ptr = thrust::device_pointer_cast(dx_esb);

			for(int k = k_start; k <= i; k++){
				if(iSe[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_e = base_index_ms_e + (iSe_ext[j]/BLOCKDIMX_COL);
					}
					sig_x_e[k] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());
					sig_y_e[k] = thrust::reduce(base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL), base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL) + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());

					sig_x_e[k] = sqrt((sig_x_e[k] - (iSe[k] * xbar_e[k] * xbar_e[k]))/iSe[k]);
					sig_y_e[k] = sqrt((sig_y_e[k] - (iSe[k] * ybar_e[k] * ybar_e[k]))/iSe[k]);

					base_index_ms_e = 0;
				}
				else{
					sig_x_e[k] = 0;
					sig_y_e[k] = 0;
				}
			}

		        ptr = thrust::device_pointer_cast(dx_es);

        		for(int k = k_start; k <= i; k++)
        		{
                		if(iSe[k] > 1)
                		{
                        		for(int j = k - 1; j >= 0; j--)
                        		{
                                		base_index_ms_e = base_index_ms_e + iSe_ext[j];
                        		}


                        		xpbar_e[k] = thrust::reduce(base_index_ms_e + ptr + Npart_e_ext, base_index_ms_e + ptr + Npart_e_ext + iSe[k], (double) 0, thrust::plus<double>())/iSe[k];
                        		ypbar_e[k] = thrust::reduce(base_index_ms_e + ptr + (3 * Npart_e_ext), base_index_ms_e + ptr + (3 * Npart_e_ext) + iSe[k], (double) 0, thrust::plus<double>())/iSe[k];

                        		base_index_ms_e = 0;
                		}
                		else
                		{
                        		xpbar_e[k] = 0;
                        		ypbar_e[k] = 0;
                		}

        		}

		}
		
		if(p_bunch != -1){
			numBlocks = (ppart)/numThreads + (((ppart)%numThreads)?1:0);
			//cout<<"Before kick P:"<<numBlocks<<","<<numThreads<<endl;

                        applyKickGPU_P<<<numBlocks, numThreads>>>(dxpbar_p, dypbar_p, dxbar_e, dybar_e, d_sig_x_e, d_sig_y_e, d_iSe_rcvd, bParams->N_e, dx_ps, bParams->gamma_p, Npart_p_ext, &dS_p[S_Index], i+1, ppart, 0, dx_pmb, dx_psb, 1, Npart_inbound_e_rcvd, bParams->N);
			
			cudaDeviceSynchronize();
			
			ptr = thrust::device_pointer_cast(dx_pmb);

			for(int k = k_start; k <= i; k++){
				if(iSp[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_p = base_index_ms_p + (iSp_ext[j]/BLOCKDIMX_COL);
					}
					xbar_p[k] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSp[k];
					ybar_p[k] = thrust::reduce(base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr, base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSp[k];

					base_index_ms_p = 0;
				}
				else{
					xbar_p[k] = 0;
					ybar_p[k] = 0;
				}
			}
			
			ptr = thrust::device_pointer_cast(dx_psb);

			for(int k = k_start; k <= i; k++){
				if(iSp[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_p = base_index_ms_p + (iSp_ext[j]/BLOCKDIMX_COL);
					}
					sig_x_p[k] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());
					sig_y_p[k] = thrust::reduce(base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr, base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());

					sig_x_p[k] = sqrt((sig_x_p[k] - (iSp[k] * xbar_p[k] * xbar_p[k]))/iSp[k]);
					sig_y_p[k] = sqrt((sig_y_p[k] - (iSp[k] * ybar_p[k] * ybar_p[k]))/iSp[k]);

					base_index_ms_p = 0;
				}
				else{
					sig_x_p[k] = 0;
					sig_y_p[k] = 0;
				}
			}

		        ptr = thrust::device_pointer_cast(dx_ps);

        		for(int k = k_start; k <= i; k++)
        		{
                		if(iSp[k] > 1)
                		{
                        		for(int j = k - 1; j >= 0; j--)
                        		{
                                		base_index_ms_p = base_index_ms_p + iSp_ext[j];
                        		}

		                        xpbar_p[k] = thrust::reduce(base_index_ms_p + ptr + Npart_p_ext, base_index_ms_p + ptr + Npart_p_ext + iSp[k], (double) 0, thrust::plus<double>())/iSp[k];
                		        ypbar_p[k] = thrust::reduce(base_index_ms_p + (3 * Npart_p_ext) + ptr, base_index_ms_p + (3 * Npart_p_ext) + ptr + iSp[k], (double) 0, thrust::plus<double>())/iSp[k];

                        		base_index_ms_p = 0;
                		}
                		else
                		{
                        		xpbar_p[k] = 0;
                        		ypbar_p[k] = 0;
                		}
        		}

		}

		//quad::timer::start_timer(&timer_comm2);
		
		if(e_bunch != -1){
		
			if(opp_p_bunch_gpu != world_rank){
				//send e-beam mean and sd to opp_p_bunch GPU
				//receieve p-beam mean and sd from opp_p_bunch GPU
				//receieve iSp from opp_p_bunch GPU
				//receieve Npart_ibound_p from opp_p_bunch GPU
				
				MPI_Isend(xbar_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_send_e[0]);
				MPI_Isend(ybar_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_send_e[1]);
				MPI_Isend(sig_x_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 3, MPI_COMM_WORLD, &myRequest_send_e[2]);
				MPI_Isend(sig_y_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_send_e[3]);
				
				MPI_Irecv(xbar_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_recv_e[0]);
				MPI_Irecv(ybar_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_recv_e[1]);
				MPI_Irecv(sig_x_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_recv_e[2]);
				MPI_Irecv(sig_y_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_recv_e[3]);
			}
			else{
				memcpy(xbar_p_rcvd, xbar_p, sizeof(double) * N);
				memcpy(ybar_p_rcvd, ybar_p, sizeof(double) * N);
				memcpy(sig_x_p_rcvd, sig_x_p, sizeof(double) * N);
				memcpy(sig_y_p_rcvd, sig_y_p, sizeof(double) * N);
			}
		}
	
		if(p_bunch != -1){
		
			if(opp_e_bunch_gpu != world_rank){
				//send p-beam mean and sd to opp_e_bunch GPU
				//receieve e-beam mean and sd from opp_e_bunch GPU
				//store receieved in xbar_e_rcvd, sig_x_e_rcvd
				//receieve iSe from opp_e_bunch GPU
				//receieve Npart_ibound_e from opp_e_bunch GPU
				
				MPI_Isend(xbar_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_send_p[0]);
				MPI_Isend(ybar_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_send_p[1]);
				MPI_Isend(sig_x_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_send_p[2]);
				MPI_Isend(sig_y_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_send_p[3]);
				
				MPI_Irecv(xbar_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_recv_p[0]);
				MPI_Irecv(ybar_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_recv_p[1]);
				MPI_Irecv(sig_x_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 3, MPI_COMM_WORLD, &myRequest_recv_p[2]);
				MPI_Irecv(sig_y_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_recv_p[3]);
			}
			else{
				memcpy(xbar_e_rcvd, xbar_e, sizeof(double) * N);
				memcpy(ybar_e_rcvd, ybar_e, sizeof(double) * N);
				memcpy(sig_x_e_rcvd, sig_x_e, sizeof(double) * N);
				memcpy(sig_y_e_rcvd, sig_y_e, sizeof(double) * N);
			}
		}
	
	//DO MPI_Wait calls here
	
	        if(e_bunch != -1){
        	        if(opp_p_bunch_gpu != world_rank){
                	        MPI_Waitall(4, myRequest_send_e, status_send_e);
                        	MPI_Waitall(4, myRequest_recv_e, status_recv_e);
                	}
        	}

	        if(p_bunch != -1){
        	        if(opp_e_bunch_gpu != world_rank){
                	        MPI_Waitall(4, myRequest_send_p, status_send_p);
                        	MPI_Waitall(4, myRequest_recv_p, status_recv_p);
	                }
        	}

		//quad::timer::stop_timer(&timer_comm2, "Comm2");
	
		if(e_bunch != -1){
			(cudaMemcpy(dxbar_p, xbar_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(dybar_p, ybar_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));

                	(cudaMemcpy(dxpbar_e, xpbar_e, sizeof(double) * N, cudaMemcpyHostToDevice));
                	(cudaMemcpy(dypbar_e, ypbar_e, sizeof(double) * N, cudaMemcpyHostToDevice));
			
			(cudaMemcpy(d_sig_x_p, sig_x_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(d_sig_y_p, sig_y_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		}
		
		if(p_bunch != -1){
			(cudaMemcpy(dxbar_e, xbar_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(dybar_e, ybar_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));

  			(cudaMemcpy(dxpbar_p, xpbar_p, sizeof(double) * N, cudaMemcpyHostToDevice));
  			(cudaMemcpy(dypbar_p, ypbar_p, sizeof(double) * N, cudaMemcpyHostToDevice));

			(cudaMemcpy(d_sig_x_e, sig_x_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(d_sig_y_e, sig_y_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		}

		S_Index = S_Index + (i + 1);
	}

	calculate_rms(lum_turn, e_bunch, p_bunch, dx_es, dx_ps, N, iSe, iSp, iSe_ext, iSp_ext, Npart_e_ext, Npart_p_ext, 1);
	
	for(int i = 1; i <= N-1; i++){
	
		epart = 0; ppart = 0;
	
		for(int j = i; j <= N-1; j++){
			if(e_bunch != -1)
				epart += iSe_ext[j];
			if(p_bunch != -1)
				ppart += iSp_ext[j];
			if(e_bunch != -1)
			{	
				//cout<<"doing e dump vars 2"<<endl;
				xbar_e_dump[col_num_dump*N + j] = xbar_e[j];
				ybar_e_dump[col_num_dump*N + j] = ybar_e[j];
				sig_x_e_dump[col_num_dump*N + j] = sig_x_e[j];
				sig_y_e_dump[col_num_dump*N + j] = sig_y_e[j];
				if(p_bunch == -1)
				{
					//cout<<"doing p dump vars 2, from e"<<endl;
					xbar_p_dump[col_num_dump*N + j] = xbar_p_rcvd[j];
					ybar_p_dump[col_num_dump*N + j] = ybar_p_rcvd[j];
					sig_x_p_dump[col_num_dump*N + j] = sig_x_p_rcvd[j];
					sig_y_p_dump[col_num_dump*N + j] = sig_y_p_rcvd[j];
				}
			}
			
			if(p_bunch != -1)
			{//crashes here
				//cout<<"doing p dump vars 2"<<endl;
				if(e_bunch == -1)
				{
					//cout<<"doing e dump vars 2, from p"<<endl;
					xbar_e_dump[col_num_dump*N + j] = xbar_e_rcvd[j];
					ybar_e_dump[col_num_dump*N + j] = ybar_e_rcvd[j];
					sig_x_e_dump[col_num_dump*N + j] = sig_x_e_rcvd[j];
					sig_y_e_dump[col_num_dump*N + j] = sig_y_e_rcvd[j];
				}
		
				xbar_p_dump[col_num_dump*N + j] = xbar_p[j];
				ybar_p_dump[col_num_dump*N + j] = ybar_p[j];
				sig_x_p_dump[col_num_dump*N + j] = sig_x_p[j];
				sig_y_p_dump[col_num_dump*N + j] = sig_y_p[j];	
			}
		}
		col_num_dump++;
		if(e_bunch != -1){
			numBlocks = (epart)/numThreads + (((epart)%numThreads)?1:0);
			//cout<<"Before kick2 E:"<<numBlocks<<","<<numThreads<<endl;

			applyKickGPU_E<<<numBlocks, numThreads>>>(dx_es, bParams->gamma_e, Npart_e_ext, dxpbar_e, dypbar_e, dxbar_p, dybar_p, d_sig_x_p, d_sig_y_p, d_iSp_rcvd, bParams->N_p, &dS_e[S_Index], bParams->N - i, epart, i, dx_emb, dx_esb, 0, Npart_inbound_p_rcvd, bParams->N);
			
			cudaDeviceSynchronize();
			
			ptr = thrust::device_pointer_cast(dx_emb);

			for(int k = i+1; k < N; k++){
				if(iSe[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_e = base_index_ms_e + (iSe_ext[j]/BLOCKDIMX_COL);
					}
					xbar_e[k] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSe[k];
					ybar_e[k] = thrust::reduce(base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL), base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL) + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSe[k];

					base_index_ms_e = 0;
				}
				else{
					xbar_e[k] = 0;
					ybar_e[k] = 0;
				}
			}
			
			ptr = thrust::device_pointer_cast(dx_esb);

			for(int k = i+1; k < N; k++){
				if(iSe[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_e = base_index_ms_e + (iSe_ext[j]/BLOCKDIMX_COL);
					}
					sig_x_e[k] = thrust::reduce(base_index_ms_e + ptr, base_index_ms_e + ptr + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());
					sig_y_e[k] = thrust::reduce(base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL), base_index_ms_e + ptr + (Npart_e_ext/BLOCKDIMX_COL) + (iSe_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());

					sig_x_e[k] = sqrt((sig_x_e[k] - (iSe[k] * xbar_e[k] * xbar_e[k]))/iSe[k]);
					sig_y_e[k] = sqrt((sig_y_e[k] - (iSe[k] * ybar_e[k] * ybar_e[k]))/iSe[k]);

					base_index_ms_e = 0;
				}
				else{
					sig_x_e[k] = 0;
					sig_y_e[k] = 0;
				}
			}

		        ptr = thrust::device_pointer_cast(dx_es);

        		for(int k = i+1; k < N; k++)
        		{
                		if(iSe[k] > 1)
                		{
                        		for(int j = k - 1; j >= 0; j--)
                        		{
                                		base_index_ms_e = base_index_ms_e + iSe_ext[j];
                        		}

               			        xpbar_e[k] = thrust::reduce(base_index_ms_e + ptr + Npart_e_ext, base_index_ms_e + ptr + Npart_e_ext + iSe[k], (double) 0, thrust::plus<double>())/iSe[k];
		                        ypbar_e[k] = thrust::reduce(base_index_ms_e + ptr + (3 * Npart_e_ext), base_index_ms_e + ptr + (3 * Npart_e_ext) + iSe[k], (double) 0, thrust::plus<double>())/iSe[k];

                		        base_index_ms_e = 0;
                		}
                		else
                		{
                        		xpbar_e[k] = 0;
                        		ypbar_e[k] = 0;
                		}

        		}

		}
		
		if(p_bunch != -1){
			numBlocks = (ppart)/numThreads + (((ppart)%numThreads)?1:0);
			//cout<<"Before Kick2 P:"<<numBlocks<<","<<numThreads<<endl;

			applyKickGPU_P<<<numBlocks, numThreads>>>(dxpbar_p, dypbar_p, dxbar_e, dybar_e, d_sig_x_e, d_sig_y_e, d_iSe_rcvd, bParams->N_e, dx_ps, bParams->gamma_p, Npart_p_ext, &dS_p[S_Index], bParams->N - i, ppart, i, dx_pmb, dx_psb, 0, Npart_inbound_e_rcvd, bParams->N);
			
			cudaDeviceSynchronize();
			
			ptr = thrust::device_pointer_cast(dx_pmb);

			for(int k = i+1; k < N; k++){
				if(iSp[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_p = base_index_ms_p + (iSp_ext[j]/BLOCKDIMX_COL);
					}
					xbar_p[k] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSp[k];
					ybar_p[k] = thrust::reduce(base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr, base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>())/iSp[k];

					base_index_ms_p = 0;
				}
				else{
					xbar_p[k] = 0;
					ybar_p[k] = 0;
				}
			}
			
			ptr = thrust::device_pointer_cast(dx_psb);

			for(int k = i+1; k < N; k++){
				if(iSp[k] > 1){
					for(int j = k - 1; j >= 0; j--){
							base_index_ms_p = base_index_ms_p + (iSp_ext[j]/BLOCKDIMX_COL);
					}
					sig_x_p[k] = thrust::reduce(base_index_ms_p + ptr, base_index_ms_p + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());
					sig_y_p[k] = thrust::reduce(base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr, base_index_ms_p + (Npart_p_ext/BLOCKDIMX_COL) + ptr + (iSp_ext[k]/BLOCKDIMX_COL), (double) 0, thrust::plus<double>());

					sig_x_p[k] = sqrt((sig_x_p[k] - (iSp[k] * xbar_p[k] * xbar_p[k]))/iSp[k]);
					sig_y_p[k] = sqrt((sig_y_p[k] - (iSp[k] * ybar_p[k] * ybar_p[k]))/iSp[k]);

					base_index_ms_p = 0;
				}
				else{
					sig_x_p[k] = 0;
					sig_y_p[k] = 0;
				}
			}

        		ptr = thrust::device_pointer_cast(dx_ps);

		        for(int k = i+1; k < N; k++)
        		{
                		if(iSp[k] > 1)
                		{
                        		for(int j = k - 1; j >= 0; j--)
                        		{
                                		base_index_ms_p = base_index_ms_p + iSp_ext[j];
                        		}

		                        xpbar_p[k] = thrust::reduce(base_index_ms_p + ptr + Npart_p_ext, base_index_ms_p + ptr + Npart_p_ext + iSp[k], (double) 0, thrust::plus<double>())/iSp[k];
                		        ypbar_p[k] = thrust::reduce(base_index_ms_p + (3 * Npart_p_ext) + ptr, base_index_ms_p + (3 * Npart_p_ext) + ptr + iSp[k], (double) 0, thrust::plus<double>())/iSp[k];

		                        base_index_ms_p = 0;
                		}
                		else
                		{
                        		xpbar_p[k] = 0;
                        		ypbar_p[k] = 0;
                		}
        		}
		}

		//quad::timer::start_timer(&timer_comm2);
		
		if(e_bunch != -1){
		
			if(opp_p_bunch_gpu != world_rank){
				//send e-beam mean and sd to opp_p_bunch GPU
				//receieve p-beam mean and sd from opp_p_bunch GPU
				//receieve iSp from opp_p_bunch GPU
				//receieve Npart_ibound_p from opp_p_bunch GPU
				
				MPI_Isend(xbar_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_send_e[0]);
				MPI_Isend(ybar_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_send_e[1]);
				MPI_Isend(sig_x_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 3, MPI_COMM_WORLD, &myRequest_send_e[2]);
				MPI_Isend(sig_y_e, N, MPI_DOUBLE, opp_p_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_send_e[3]);
				
				MPI_Irecv(xbar_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_recv_e[0]);
				MPI_Irecv(ybar_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_recv_e[1]);
				MPI_Irecv(sig_x_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_recv_e[2]);
				MPI_Irecv(sig_y_p_rcvd, N, MPI_DOUBLE, opp_p_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_recv_e[3]);
			}
			else{
				memcpy(xbar_p_rcvd, xbar_p, sizeof(double) * N);
				memcpy(ybar_p_rcvd, ybar_p, sizeof(double) * N);
				memcpy(sig_x_p_rcvd, sig_x_p, sizeof(double) * N);
				memcpy(sig_y_p_rcvd, sig_y_p, sizeof(double) * N);
			}
		}
	
		if(p_bunch != -1){
		
			if(opp_e_bunch_gpu != world_rank){
				//send p-beam mean and sd to opp_e_bunch GPU
				//receieve e-beam mean and sd from opp_e_bunch GPU
				//store receieved in xbar_e_rcvd, sig_x_e_rcvd
				//receieve iSe from opp_e_bunch GPU
				//receieve Npart_ibound_e from opp_e_bunch GPU
				
				MPI_Isend(xbar_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_send_p[0]);
				MPI_Isend(ybar_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_send_p[1]);
				MPI_Isend(sig_x_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_send_p[2]);
				MPI_Isend(sig_y_p, N, MPI_DOUBLE, opp_e_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_send_p[3]);
				
				MPI_Irecv(xbar_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_recv_p[0]);
				MPI_Irecv(ybar_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_recv_p[1]);
				MPI_Irecv(sig_x_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 3, MPI_COMM_WORLD, &myRequest_recv_p[2]);
				MPI_Irecv(sig_y_e_rcvd, N, MPI_DOUBLE, opp_e_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_recv_p[3]);
			}
			else{
				memcpy(xbar_e_rcvd, xbar_e, sizeof(double) * N);
				memcpy(ybar_e_rcvd, ybar_e, sizeof(double) * N);
				memcpy(sig_x_e_rcvd, sig_x_e, sizeof(double) * N);
				memcpy(sig_y_e_rcvd, sig_y_e, sizeof(double) * N);
			}
		}
	
	//DO MPI_Wait calls here
	
                if(e_bunch != -1){
                        if(opp_p_bunch_gpu != world_rank){
                                MPI_Waitall(4, myRequest_send_e, status_send_e);
                                MPI_Waitall(4, myRequest_recv_e, status_recv_e);
                        }
                }

                if(p_bunch != -1){
                        if(opp_e_bunch_gpu != world_rank){
                                MPI_Waitall(4, myRequest_send_p, status_send_p);
                                MPI_Waitall(4, myRequest_recv_p, status_recv_p);
                        }
                }

		//quad::timer::stop_timer(&timer_comm2, "Comm2");
	
		if(e_bunch != -1){
			(cudaMemcpy(dxbar_p, xbar_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(dybar_p, ybar_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			
			(cudaMemcpy(d_sig_x_p, sig_x_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(d_sig_y_p, sig_y_p_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));

        	        (cudaMemcpy(dxpbar_e, xpbar_e, sizeof(double) * N, cudaMemcpyHostToDevice));
	                (cudaMemcpy(dypbar_e, ypbar_e, sizeof(double) * N, cudaMemcpyHostToDevice));

		}
		
		if(p_bunch != -1){
			(cudaMemcpy(dxbar_e, xbar_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(dybar_e, ybar_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));

  			(cudaMemcpy(dxpbar_p, xpbar_p, sizeof(double) * N, cudaMemcpyHostToDevice));
  			(cudaMemcpy(dypbar_p, ypbar_p, sizeof(double) * N, cudaMemcpyHostToDevice));

			(cudaMemcpy(d_sig_x_e, sig_x_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
			(cudaMemcpy(d_sig_y_e, sig_y_e_rcvd, sizeof(double) * N, cudaMemcpyHostToDevice));
		}
		
		S_Index = S_Index + bParams->N - i;

	}

	calculate_rms(lum_turn, e_bunch, p_bunch, dx_es, dx_ps, N, iSe, iSp, iSe_ext, iSp_ext, Npart_e_ext, Npart_p_ext, 2);


	numBlocks = Npart_e_ext/numThreads + ((Npart_e_ext%numThreads)?1:0);
	merge<<<numBlocks, numThreads>>>(dx_e, dx_es, bParams->Npart_e, Npart_e_ext);

	numBlocks = Npart_p_ext/numThreads + ((Npart_p_ext%numThreads)?1:0);
	merge<<<numBlocks, numThreads>>>(dx_p, dx_ps, bParams->Npart_p, Npart_p_ext);

	cudaDeviceSynchronize();
	//cout<<"About to dumpSlices:"<<world_rank<<endl;
        //cout<<"e:"<<e_bunch<<" p:"<<p_bunch<<" ("<<world_rank<<")"<<endl;;
	if(e_bunch !=-1 && p_bunch!=-1)
		dumpSlices(N, bParams->Npart_e, bParams->Npart_p, iSe, iSp, bParams, xbar_e_dump, ybar_e_dump, sig_x_e_dump, sig_y_e_dump, xbar_p_dump, ybar_p_dump, sig_x_p_dump, sig_y_p_dump);
	else if(e_bunch!=-1)
		dumpSlices(N, bParams->Npart_e, bParams->Npart_p, iSe, iSp_rcvd, bParams, xbar_e_dump, ybar_e_dump, sig_x_e_dump, sig_y_e_dump, xbar_p_dump, ybar_p_dump, sig_x_p_dump, sig_y_p_dump);
	else if(p_bunch!=-1)
		dumpSlices(N, bParams->Npart_e, bParams->Npart_p, iSe_rcvd, iSp, bParams, xbar_e_dump, ybar_e_dump, sig_x_e_dump, sig_y_e_dump, xbar_p_dump, ybar_p_dump, sig_x_p_dump, sig_y_p_dump);
	//cout<<"Dumped Slices:"<<world_rank<<endl;
        if(e_bunch != -1){
	/*	delete[] z_e;
		delete[] z_p_rcvd;
		delete[] SS_e;
		delete[] iSe;
		delete[] iSe_ext;
		delete[] xbar_e;
		delete[] ybar_e; 
		delete[] sig_x_e; 
		delete[] sig_y_e; 
		delete[] xbar_e_rcvd; 
		delete[] ybar_e_rcvd; 
		delete[] sig_x_e_rcvd; 
		delete[] sig_y_e_rcvd; 
		delete[] iSe_rcvd;
	*/
		cudaFree(dS_e);
		cudaFree(d_iSe);
		cudaFree(d_iSe_ext);
		cudaFree(d_iSe_Inc);
		cudaFree(dx_es);
		cudaFree(dx_ems);
		cudaFree(dx_esd);
		cudaFree(dx_emb);
		cudaFree(dx_esb);
		cudaFree(dxbar_p);
		cudaFree(dybar_p);
                cudaFree(dxpbar_e);
                cudaFree(dypbar_e);
		cudaFree(d_sig_x_p);
		cudaFree(d_sig_y_p);
		cudaFree(d_iSp_rcvd);
	}
	
	if(p_bunch != -1){
   
                cudaFree(dS_p);
                cudaFree(d_iSp);
                cudaFree(d_iSp_ext);
                cudaFree(d_iSp_Inc);
                cudaFree(dx_ps);
                cudaFree(dx_pms);
                cudaFree(dx_psd);
                cudaFree(dx_pmb);
                cudaFree(dx_psb);
                cudaFree(dxbar_e);
                cudaFree(dybar_e);
                cudaFree(dypbar_p);
                cudaFree(dxpbar_p);
                cudaFree(d_sig_x_e);
                cudaFree(d_sig_y_e);
                cudaFree(d_iSe_rcvd);
	}
	//cout<<"Done Collide:"<<opp_e_bunch_gpu<<"-["<<opp_p_bunch_gpu<<"]"<<endl;
}
