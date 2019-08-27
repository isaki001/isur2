#include <iostream>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <complex>
#include <math.h>    //I added this
#include "common.h"
#include "Beam.h"
#include "IO.h"
#include "util/Util.h"
#include <ctime>

using namespace std;

void Beam::initialize(BeamParams *bParams){
  int N = bParams->N;
  z_e = new double[N];
  z_p = new double[N];
  s_x_e = new double[N * (2 * N)];
  s_y_e = new double[N * (2 * N)];
  s_x_p = new double[N * (2 * N)];
  s_y_p = new double[N * (2 * N)];

  xb_e = new double[N * (2 * N)];
  yb_e = new double[N * (2 * N)];
  xb_p = new double[N * (2 * N)];
  yb_p = new double[N * (2 * N)];
  iSize_e = new int[N];
  iSize_p = new int[N];

  fmap = new Map;
  fmap->Nrow = new int[6];
  fmap->M = new double[10000 * 6];
  fmap->it = new int[10000 * 6 * 6];

  map_e = new Map;
  map_p = new Map;
  eqns_e = new Map;
  eqns_p = new Map;

  //!------------------------------------------------
  //! Compute the parameters used in the simulation. 
  //!------------------------------------------------
  bParams->E_e0 = 511e3;                 //! [eV]
  bParams->E_p0 = 938e6;                //! [eV]

  bParams->gamma_e = bParams->E_e/bParams->E_e0;
  bParams->gamma_p = bParams->E_p/bParams->E_p0;
  bParams->Lc      = 1e-04 * bParams->N_e * bParams->N_p * bParams->f_0/(2.0 * PI);         //! [cm^{-2} s^{-1}]

  x_e = new double[(bParams->ebunches) * (bParams->Npart_e * (NCOL + 2))];
  x_p = new double[bParams->pbunches * (bParams->Npart_p * (NCOL + 2))];
  
}

void Beam::allocateMap(Map *map, int len){
  map->Nrow = new int[6];
  map->M = new double[len * 6];
  map->it = new int[len * 6 * 6];  
  for(int i = 0; i < len * 6; ++i){
    map->M[i] = 0;
  }
}

void Beam::genLinMap(int e_or_p, double tunex, double tuney, double tunez, double betax, double betay, double ga, double sig_s, double sig_dE, int world_rank){
  std::_Ios_Openmode mode = std::ios::app;
  double M[6][6];
  double beta, alphax, alphay, gx, cx, sx, gy, cy, sy, cz, sz;
  double pi = 3.141592653589793238462643383;
  double twopi = 2.0*pi;

  beta = sqrt(1.0 - (1.0/pow(ga,2)));
  alphax = 0.0;
  alphay = 0.0;
  gx = (1.0+alphax*alphax)/betax;
  cx = cos(twopi*tunex);
  sx = sin(twopi*tunex);
  gy = (1.0+alphay*alphay)/betay;
  cy = cos(twopi*tuney);
  sy = sin(twopi*tuney);
  cz = cos(twopi*tunez);
  sz = sin(twopi*tunez);

  M[1][1] = cx + alphax*sx;
  M[1][2] = betax*sx;
  M[2][1] = -gx*sx;
  M[2][2] = cx - alphax*sx;
  M[3][3] = cy + alphay*sy;
  M[3][4] = betay*sy;
  M[4][3] = -gy*sy;
  M[4][4] = cy - alphay*sy;
  M[5][5] = 1.0 - pow((twopi*tunez),2);
  M[5][6] = (twopi*tunez*(pow(beta,2))*(pow(ga,2))/(pow((1.0+ga),2)))*(sig_s/sig_dE);
  M[6][5] = -(twopi*tunez*(pow((1.0+ga),2))/((pow(beta,2))*(pow(ga,2))))*(sig_dE/sig_s);
  M[6][6] = 1.0;

  std::stringstream ss;
  ss.precision(12);

  ss << "1 \t " << M[1][1] << "\t 1 \t 1 \t 0 \t 0 \t 0 \t 0 \t 0" << "\n";
  ss << "2 \t " << M[1][2] << "\t 1 \t 0 \t 1 \t 0 \t 0 \t 0 \t 0" << "\n";
  ss << "1 \t " << M[2][1] << "\t 1 \t 1 \t 0 \t 0 \t 0 \t 0 \t 0" << "\n";
  ss << "2 \t " << M[2][2] << "\t 1 \t 0 \t 1 \t 0 \t 0 \t 0 \t 0" << "\n";
  ss << "1 \t " << M[3][3] << "\t 1 \t 0 \t 0 \t 1 \t 0 \t 0 \t 0" << "\n";
  ss << "2 \t " << M[3][4] << "\t 1 \t 0 \t 0 \t 0 \t 1 \t 0 \t 0" << "\n";
  ss << "1 \t " << M[4][3] << "\t 1 \t 0 \t 0 \t 1 \t 0 \t 0 \t 0" << "\n";
  ss << "2 \t " << M[4][4] << "\t 1 \t 0 \t 0 \t 0 \t 1 \t 0 \t 0" << "\n";
  ss << "1 \t " << M[5][5] << "\t 1 \t 0 \t 0 \t 0 \t 0 \t 1 \t 0" << "\n";
  ss << "2 \t " << M[5][6] << "\t 1 \t 0 \t 0 \t 0 \t 0 \t 0 \t 1" << "\n";
  ss << "1 \t " << M[6][5] << "\t 1 \t 0 \t 0 \t 0 \t 0 \t 1 \t 0" << "\n";
  ss << "2 \t " << M[6][6] << "\t 1 \t 0 \t 0 \t 0 \t 0 \t 0 \t 1" << "\n";

  if(e_or_p == 1) {
    system("exec rm -r input/map_e.dat");
    std::ofstream file("input/map_e.dat", mode);
    if (!file.is_open()){
      std::string msg = "Cannot open map file ";
      __Abort(msg.c_str());
    }
    if(world_rank == 0)
    	file << ss.str();
    file.close();
  }
  else {
   system("exec rm -r input/map_p.dat");
   std::ofstream file("input/map_p.dat", mode);
   if (!file.is_open()){
     std::string msg = "Cannot open map file ";
     __Abort(msg.c_str());
   }
   if(world_rank == 0)
   	file << ss.str();
   file.close();
  }
}


//!==============================================================================
//! Generate initial conditions by random sampling using Gaussian distribution.
//!------------------------------------------------------------------------------
/*void Beam::genICs(BeamParams *bParams, PARTICLE_TYPE type){
  int iSeed = 199;
  std::stringstream ss;
  int Npart = 0;
  double p_0 = 0;
  if(type == PROTON){
    Npart = bParams->Npart_p;
    p_0 = sqrt(pow((bParams->sig_x0_p/bParams->beta_x0), 2.0) + pow((bParams->sig_y0_p/bParams->beta_y0), 2.0));
    int Nseed = iSeed;
    double ga = bParams->gamma_p;
   
 
     if(bParams->icGen==1){
         for(int i = 0; i < Npart; ++i){
        int horizontalCoord = bParams->coord1;
        int verticalCoord = bParams->coord2;
        double xlow = bParams->x_l;
        double xup = bParams->x_u;
        double ylow = bParams->y_l;
        double yup = bParams->y_u;
        int horizontalLen = bParams->coordNum1;
        int verticalLen = bParams->coordNum2;

        if (horizontalLen*verticalLen!=Npart){
          std::string msg = "Number of values for each coordinate must multiply to total number of particles.";
          Abort(msg.c_str());
        }
        if (horizontalLen<=1||verticalLen<=1){
          std::string msg = "Number of values for each coordinate must be greater than 1.";
          Abort(msg.c_str());
        }
        if (horizontalCoord<1||horizontalCoord>6||verticalCoord<1||verticalCoord>6){
          std::string msg = "The two coordinates must be an integer 1-6.";
          Abort(msg.c_str());
        }
        if (horizontalCoord==verticalCoord){
          std::string msg = "The two coordinates must be different.";
          Abort(msg.c_str());
        }
        if (xlow>=xup||ylow>=yup){
          std::string msg = "The upper bounds must be greater than the lower bounds.";
          Abort(msg.c_str());
        }

        int horizontalPosition = i%horizontalLen;
        int verticalPosition = i/horizontalLen;
        for(int j = 0; j < 6; ++j){
          if(j+1==horizontalCoord){
            x_p[j * Npart+i] = xlow+(xup-xlow)*(((double)horizontalPosition)/((double)(horizontalLen-1)));
          }
          else if(j+1==verticalCoord){
            x_p[j * Npart+i] = ylow+(yup-ylow)*(((double)verticalPosition)/((double)(verticalLen-1)));
          }
          else{
            x_p[j * Npart+i] = 0.0;
          }
        }
	 }
     }

     /////////////////////////////////////////////////////////////////////
else if(bParams->icGen==2){
        int horizontalCoord = bParams->coord1;
        int verticalCoord = bParams->coord2;
        double xlow = bParams->x_l;
        double xup = bParams->x_u;
        double ylow = bParams->y_l;
        double yup = bParams->y_u;
        int horizontalLen = bParams->coordNum1;
        int verticalLen = bParams->coordNum2;

        if (horizontalLen*verticalLen!=Npart){
          std::string msg = "Number of values for each coordinate must multiply to total number of particles.";
          Abort(msg.c_str());
        }
        if (horizontalLen<=1||verticalLen<=1){
          std::string msg = "Number of values for each coordinate must be greater than 1.";
          Abort(msg.c_str());
        }
        if (horizontalCoord<1||horizontalCoord>6||verticalCoord<1||verticalCoord>6){
          std::string msg = "The two coordinates must be an integer 1-6.";
          Abort(msg.c_str());
        }
        if (horizontalCoord==verticalCoord){
          std::string msg = "The two coordinates must be different.";
          Abort(msg.c_str());
        }
        if (xlow>=xup||ylow>=yup){
          std::string msg = "The upper bounds must be greater than the lower bounds.";
          Abort(msg.c_str());
        }
	
	double set= Npart/182;
	double Lx=xup-xlow;
	double Ly=yup-ylow;
	double Rmax;
	double dR;
	double DTheta;
	int index;
	double r;
	double Theta;

	if(Lx/2>=Ly){
		Rmax=Lx/2;
	}
	else {
		Rmax=Ly;
	}

	dR=Rmax/set;
	DTheta=PI/180;
	index=0;
	for (int i=0; i<=180; i++) { 
	  Theta=i*DTheta;
         for (int j=0; j<=set; j++) {   
		   r=j*dR;
                 x_p[index]=r*cos(Theta);
                 x_p[2*Npart+index]=r*sin(Theta);
                 x_p[Npart + index] = 0.0;
                 x_p[3 * Npart + index] = 0.0;
                 x_p[4 * Npart + index] = 0.0;
                 x_p[5 * Npart + index] = 0.0;
                 index=index+1;	 
         }
     }
 }
  
      else{
	for(int i = 0; i < Npart; ++i){
        x_p[i]         = Util::gauss(0.0, bParams->sig_x0_p, 3.0, Nseed);
        x_p[Npart + i] = Util::gauss(0.0, bParams->sig_x0_p/bParams->beta_x0, 3.0,Nseed);
        x_p[2 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_p, 3.0, Nseed);
        x_p[3 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_p/bParams->beta_y0, 3.0, Nseed);
        x_p[4 * Npart + i] = Util::gauss(0.0, bParams->sig_z0_p * (ga/(1.0 + ga)), 3.0,Nseed);
        x_p[5 * Npart + i] = Util::gauss(0.0, (bParams->sig_dE0/bParams->E_p)*(ga/(1.0+ga)), 3.0, Nseed);
	}
      }

  }else{
    Npart = bParams->Npart_e;
    p_0 = sqrt(pow((bParams->sig_x0_e/bParams->beta_x0), 2.0) + pow((bParams->sig_y0_e/bParams->beta_y0), 2.0));
    int Nseed = iSeed;
    std::cout << std::setprecision(16);
    std::cout << std::scientific;
    double ga = bParams->gamma_e;
      if(bParams->icGen==1){ 
        for(int i = 0; i < Npart; ++i){  
              
        int horizontalCoord = bParams->coord1;
        int verticalCoord = bParams->coord2;
        double xlow = bParams->x_l;
        double xup = bParams->x_u;
        double ylow = bParams->y_l;
        double yup = bParams->y_u;
        int horizontalLen = bParams->coordNum1;
        int verticalLen = bParams->coordNum2;

        if (horizontalLen*verticalLen!=Npart){
          std::string msg = "Number of values for each coordinate must multiply to total number of particles.";
          Abort(msg.c_str());
        }
        if (horizontalLen<=1||verticalLen<=1){
          std::string msg = "Number of values for each coordinate must be greater than 1.";
          Abort(msg.c_str());
        }
        if (horizontalCoord<1||horizontalCoord>6||verticalCoord<1||verticalCoord>6){
          std::string msg = "The two coordinates must be an integer 1-6.";
          Abort(msg.c_str());
        }
        if (horizontalCoord==verticalCoord){
          std::string msg = "The two coordinates must be different.";
          Abort(msg.c_str());
        }
        if (xlow>=xup||ylow>=yup){
          std::string msg = "The upper bounds must be greater than the lower bounds.";
          Abort(msg.c_str());
        }

        int horizontalPosition = i%horizontalLen;
        int verticalPosition = i/horizontalLen;
        for(int j = 0; j < 6; ++j){
          if(j+1==horizontalCoord){
            x_e[j * Npart+i] = xlow+(xup-xlow)*(((double)horizontalPosition)/((double)(horizontalLen-1)));
          }
          else if(j+1==verticalCoord){
            x_e[j * Npart+i] = ylow+(yup-ylow)*(((double)verticalPosition)/((double)(verticalLen-1)));
          }
          else{
            x_e[j * Npart+i] = 0.0;
             }
           } 
	}
   }                    

 else if(bParams->icGen==2){
        int horizontalCoord = bParams->coord1;
        int verticalCoord = bParams->coord2;
        double xlow = bParams->x_l;
        double xup = bParams->x_u;
        double ylow = bParams->y_l;
        double yup = bParams->y_u;
        int horizontalLen = bParams->coordNum1;
        int verticalLen = bParams->coordNum2;

        if (horizontalLen*verticalLen!=Npart){
          std::string msg = "Number of values for each coordinate must multiply to total number of particles.";
          Abort(msg.c_str());
        }
        if (horizontalLen<=1||verticalLen<=1){
          std::string msg = "Number of values for each coordinate must be greater than 1.";
          Abort(msg.c_str());
        }
        if (horizontalCoord<1||horizontalCoord>6||verticalCoord<1||verticalCoord>6){
          std::string msg = "The two coordinates must be an integer 1-6.";
          Abort(msg.c_str());
        }
        if (horizontalCoord==verticalCoord){
          std::string msg = "The two coordinates must be different.";
          Abort(msg.c_str());
        }
        if (xlow>=xup||ylow>=yup){
          std::string msg = "The upper bounds must be greater than the lower bounds.";
          Abort(msg.c_str());
        }
	
	double set= Npart/182;
	double Lx=xup-xlow;
	double Ly=yup-ylow;
	double Rmax;
	double dR;
	double DTheta;
	int index;
	double r;
	double Theta;

	if(Lx/2>=Ly){
		Rmax=Lx/2;
	}
	else {
		Rmax=Ly;
	}

	dR=Rmax/set;
	DTheta=PI/180;
	index=0;
	for (int i=0; i<=180; i++) { 
	  Theta=i*DTheta;
         for (int j=0; j<=set; j++) {   
		   r=j*dR;
                 x_e[index]=r*cos(Theta);
                 x_e[2*Npart+index]=r*sin(Theta);
                 x_e[Npart + index] = 0.0;
                 x_e[3 * Npart + index] = 0.0;
                 x_e[4 * Npart + index] = 0.0;
                 x_e[5 * Npart + index] = 0.0;
                 index=index+1;	 
         }
     }
 }
////////////////////////////////////////////////////////////////////////////////
      else{
  for(int i = 0; i < Npart; ++i){
        x_e[i]         = Util::gauss(0.0, bParams->sig_x0_e, 3.0, Nseed);
        x_e[Npart + i] = Util::gauss(0.0, bParams->sig_x0_e/bParams->beta_x0, 3.0,Nseed);
        x_e[2 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_e, 3.0, Nseed);
        x_e[3 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_e/bParams->beta_y0, 3.0, Nseed);
        x_e[4 * Npart + i] = Util::gauss(0.0, bParams->sig_z0_e * (ga/(1.0 + ga)), 3.0,Nseed);
        x_e[5 * Npart + i] = Util::gauss(0.0, (bParams->sig_dE0/bParams->E_e)*(ga/(1.0+ga)), 3.0, Nseed);
      }
    }
  }
}*/

void Beam::genICs(BeamParams *bParams, PARTICLE_TYPE type)
{
  int iSeed = 300;
  //std::cout<<"Number of bunchess are : "<<bParams->pbunches<<"\n";
  int pbunches = bParams->pbunches;
  int offset = 0;
  std::stringstream ss;
  int Npart = 0;
  double p_0 = 0;
  double xbar = 0.0, ybar = 0.0;
  
  if(type == PROTON)
  {
    Npart = bParams->Npart_p;
    p_0 = sqrt(pow((bParams->sig_x0_p/bParams->beta_x0_p), 2.0) + pow((bParams->sig_y0_p/bParams->beta_y0_p), 2.0));
    int Nseed = iSeed;
    double ga = bParams->gamma_p;
		
    for(int bunch = 0; bunch < pbunches; bunch++)
    {
      offset = bunch * (Npart * 8);
      xbar = 0.0;
      ybar = 0.0;
      for(int i = 0; i < Npart; ++i)
      {
        x_p[offset + i]         = Util::gauss(0.0, bParams->sig_x0_p, 3.0, Nseed);
        x_p[offset + Npart + i] = Util::gauss(0.0, bParams->sig_x0_p/bParams->beta_x0_p, 3.0,Nseed);
        x_p[offset + 2 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_p, 3.0, Nseed);
        x_p[offset + 3 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_p/bParams->beta_y0_p, 3.0, Nseed);
        x_p[offset + 4 * Npart + i] = Util::gauss(0.0, bParams->sig_z0_p*(ga/(1.0+ga)) , 3.0,Nseed);
        x_p[offset + 5 * Npart + i] = Util::gauss(0.0, (bParams->sig_dE0p/bParams->E_p)*(ga/(1.0+ga)), 3.0, Nseed);

        xbar += x_p[offset + i];
        ybar += x_p[offset + 2 * Npart + i];

        xbar = xbar/Npart;
        ybar = ybar/Npart;
      }
      for(int i = 0; i < Npart; ++i)
        {
          x_p[offset + i]         = x_p[offset + i] - xbar + bParams->off_x_p;
          x_p[offset + 2 * Npart + i] = x_p[offset + 2 * Npart + i] - ybar + bParams->off_y_p;
        }
    }
  }
  else
    {
      Npart = bParams->Npart_e;
      p_0 = sqrt(pow((bParams->sig_x0_e/bParams->beta_x0_e), 2.0) + pow((bParams->sig_y0_e/bParams->beta_y0_e), 2.0));
      int Nseed = iSeed;
      std::cout << std::setprecision(16);
      std::cout << std::scientific;
      double ga = bParams->gamma_e;
      //cout<<"Populating the following x_e indices"<<endl;
      for(int bunch = 0; bunch < bParams->ebunches; bunch++)
        {
          offset = bunch * (Npart * 8);
          xbar = 0.0;
          ybar = 0.0;
          for(int i = 0; i < Npart; ++i)
            {
              x_e[offset + i]         = Util::gauss(0.0, bParams->sig_x0_e, 3.0, Nseed);
              x_e[offset + Npart + i] = Util::gauss(0.0, bParams->sig_x0_e/bParams->beta_x0_e, 3.0,Nseed);
              //if((offset + Npart + i)==10000)
              {
                //cout<<"HERE:"<<x_e[10000]<<endl;
              }
              x_e[offset + 2 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_e, 3.0, Nseed);
              x_e[offset + 3 * Npart + i] = Util::gauss(0.0, bParams->sig_y0_e/bParams->beta_y0_e, 3.0, Nseed);
              x_e[offset + 4 * Npart + i] = Util::gauss(0.0, bParams->sig_z0_e , 3.0,Nseed);
              x_e[offset + 5 * Npart + i] = Util::gauss(0.0, (bParams->sig_dE0e), 3.0, Nseed);
              //cout<<offset + i<<endl;
              //cout<<offset + Npart + i<<endl;
              //cout<<offset + 2 * Npart + i<<endl;
              //cout<<offset + 3 * Npart + i<<endl;
              //cout<<offset + 4 * Npart + i<<endl;
              //cout<<offset + 5 * Npart + i<<endl;
              xbar += x_e[offset + i];
              ybar += x_e[offset + 2 * Npart + i];

              xbar = xbar/Npart;
              ybar = ybar/Npart;
            }
			
          for(int i = 0; i < Npart; ++i)
            {	
              x_e[offset + i]         = x_e[offset + i] - xbar + bParams->off_x_e;
              x_e[offset + 2 * Npart + i] = x_e[offset + 2 * Npart + i] - ybar + bParams->off_y_e;
            }
        }
    }
	
}


void Beam::applyM(double *M, int *ind, double *x, int *Nrow, int maxLen, int Npart, int Norder, int Niter){
  int Ncol = 6;
  
  clock_t startTime = clock();

  for(int ie = 0; ie < Npart; ++ie){
    double x01 = x[ie];
    double x11 = x[Npart + ie];
    double x21 = x[Npart * 2 + ie];
    double x31 = x[Npart * 3 + ie];
    double x41 = x[Npart * 4 + ie];
    double x51 = x[Npart * 5 + ie];

    for (int iter = 0; iter < Niter; iter++){
      double x0 = x01;
      double x1 = x11;
      double x2 = x21;
      double x3 = x31;
      double x4 = x41;
      double x5 = x51;
      double xterm, mxterm;
      
      int i = 0;
      mxterm = 0;
      for (int j = 0; j < Nrow[i]; j++){
	xterm = pow(x0, ind[j * Ncol * Ncol + i * Ncol])*
	  pow(x1, ind[j * Ncol * Ncol + i * Ncol + 1]) *
	  pow(x2, ind[j * Ncol * Ncol + i * Ncol + 2]) *
	  pow(x3, ind[j * Ncol * Ncol + i * Ncol + 3]) *
	  pow(x4, ind[j * Ncol * Ncol + i * Ncol + 4]) *
	  pow(x5, ind[j * Ncol * Ncol + i * Ncol + 5]);
	mxterm = mxterm + xterm * M[j * Ncol + i];
	
	//for(int ii = 0; ii < Ncol; ++ii)
	//std::cout << ind[j * Ncol * Ncol + i * Ncol + ii] << "\t";
	//std::cout << M[j * Ncol + i] << "\n";
      }
      //exit(1);
      x01 = mxterm;
      //std::cout << x01 << "\n";
      //exit(1);
      i = 1;
      mxterm = 0;
      for (int j = 0; j < Nrow[i]; j++){
	xterm = pow(x0, ind[j * Ncol * Ncol + i * Ncol])*
	  pow(x1, ind[j * Ncol * Ncol + i * Ncol + 1]) *
	  pow(x2, ind[j * Ncol * Ncol + i * Ncol + 2]) *
	  pow(x3, ind[j * Ncol * Ncol + i * Ncol + 3]) *
	  pow(x4, ind[j * Ncol * Ncol + i * Ncol + 4]) *
	  pow(x5, ind[j * Ncol * Ncol + i * Ncol + 5]);
	mxterm = mxterm + xterm * M[j * Ncol + i];
      }
      x11 = mxterm;

      i = 2;
      mxterm = 0;
      for (int j = 0; j < Nrow[i]; j++){
	xterm = pow(x0, ind[j * Ncol * Ncol + i * Ncol])*
	  pow(x1, ind[j * Ncol * Ncol + i * Ncol + 1]) *
	  pow(x2, ind[j * Ncol * Ncol + i * Ncol + 2]) *
	  pow(x3, ind[j * Ncol * Ncol + i * Ncol + 3]) *
	  pow(x4, ind[j * Ncol * Ncol + i * Ncol + 4]) *
	  pow(x5, ind[j * Ncol * Ncol + i * Ncol + 5]);
	mxterm = mxterm + xterm * M[j * Ncol + i];
      }
      x21 = mxterm;

      i = 3;
      mxterm = 0;
      for (int j = 0; j < Nrow[i]; j++){
	xterm = pow(x0, ind[j * Ncol * Ncol + i * Ncol])*
	  pow(x1, ind[j * Ncol * Ncol + i * Ncol + 1]) *
	  pow(x2, ind[j * Ncol * Ncol + i * Ncol + 2]) *
	  pow(x3, ind[j * Ncol * Ncol + i * Ncol + 3]) *
	  pow(x4, ind[j * Ncol * Ncol + i * Ncol + 4]) *
	  pow(x5, ind[j * Ncol * Ncol + i * Ncol + 5]);
	mxterm = mxterm + xterm * M[j * Ncol + i];
      }
      x31 = mxterm;
      
      i = 4;
      mxterm = 0;
      for (int j = 0; j < Nrow[i]; j++){
	xterm = pow(x0, ind[j * Ncol * Ncol + i * Ncol])*
	  pow(x1, ind[j * Ncol * Ncol + i * Ncol + 1]) *
	  pow(x2, ind[j * Ncol * Ncol + i * Ncol + 2]) *
	  pow(x3, ind[j * Ncol * Ncol + i * Ncol + 3]) *
	  pow(x4, ind[j * Ncol * Ncol + i * Ncol + 4]) *
	  pow(x5, ind[j * Ncol * Ncol + i * Ncol + 5]);
	mxterm = mxterm + xterm * M[j * Ncol + i];
      }
      x41 = mxterm;

      i = 5;
      mxterm = 0;
      for (int j = 0; j < Nrow[i]; j++){
	xterm = pow(x0, ind[j * Ncol * Ncol + i * Ncol])*
	  pow(x1, ind[j * Ncol * Ncol + i * Ncol + 1]) *
	  pow(x2, ind[j * Ncol * Ncol + i * Ncol + 2]) *
	  pow(x3, ind[j * Ncol * Ncol + i * Ncol + 3]) *
	  pow(x4, ind[j * Ncol * Ncol + i * Ncol + 4]) *
	  pow(x5, ind[j * Ncol * Ncol + i * Ncol + 5]);
	mxterm = mxterm + xterm * M[j * Ncol + i];
      }
      x51 = mxterm;
      /*
      */
    }
    x[ie] = x01;
    x[Npart + ie] = x11;
    x[Npart * 2 + ie] = x21;
    x[Npart * 3 + ie] = x31;
    x[Npart * 4 + ie] = x41;
    x[Npart * 5 + ie] = x51;

  }
  
  //std::cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;
  //exit(1);
}

void
Beam::gf2Eqns(double *f, double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder){
  int dim = 3;
  for(int ie = 0; ie < Npart; ++ie){
    f[ie] = xi[ie];
    f[Npart + ie] = xf[Npart + ie];
    
    f[2*Npart + ie] = xi[2*Npart + ie];
    f[3*Npart + ie] = xf[3*Npart + ie];
    
    f[4*Npart + ie] = xi[4*Npart + ie];
    f[5*Npart + ie] = xf[5*Npart + ie];
  }
  
  applyM(M, ind, f, Nrow, maxLen, Npart, Norder);
  
  for(int ie = 0; ie < Npart; ++ie){
    f[ie] = xf[ie] - f[ie];
    f[Npart + ie] = xi[Npart + ie] - f[Npart + ie];
    
    f[2 * Npart + ie] = xf[2 * Npart + ie] - f[2 * Npart + ie];
    f[3 * Npart + ie] = xi[3 * Npart + ie] - f[3 * Npart + ie];
    
    f[4 * Npart + ie] = xf[4 * Npart + ie] - f[4 * Npart + ie];
    f[5 * Npart + ie] = xi[5 * Npart + ie] - f[5 * Npart + ie];
  }
}

void
Beam::Jacobian(int *flag, double *jac, double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder){
  int Ncol = 6;
  double *fi = new double[Npart * Ncol];
  gf2Eqns(fi, M, ind, xi, xf, Nrow, maxLen, Npart, Norder);

  double *x = new double[Npart * Ncol];
  memcpy(x, xf, sizeof(double) * Npart * Ncol);


  //double precision = 2.2e-16;
  double *delta = new double[Npart];
  for(int j = 0; j < 6; ++j){
    for(int ie = 0; ie < Npart; ++ie){
      if(flag[ie] == 1)continue; //Skip the particle
      delta[ie] = sqrt(PRECISION*fabs(x[j * Npart + ie]));
      if (delta[ie]==0) delta[ie]=PRECISION;
      x[j * Npart + ie] += delta[ie];      
    } 
    double *ff = new double[Npart * Ncol];
    gf2Eqns(ff, M, ind, xi, x, Nrow, maxLen, Npart, Norder);
    for(int ie = 0; ie < Npart; ++ie){
      for(int i = 0; i < 6; ++i){
	if(flag[ie] == 1){
	  jac[j * 6 * Npart + i * Npart + ie] = 1;
	}else{
	  jac[j * 6 * Npart + i * Npart + ie] =  (ff[i * Npart + ie] - fi[i * Npart + ie])/delta[ie];
	}
	//if(ie == 0){
	//std::cout << "jac\t" << jac[j * 6 * Npart + i * Npart + ie] << "\t" << ff[i * Npart + ie] << "\t" << fi[i * Npart + ie] << "\t" << delta[ie] << "\n";
	//}

      }
    }    
    for(int ie = 0; ie < Npart; ++ie){
      x[j * Npart + ie] = xf[j * Npart + ie];
    }
    delete ff;
  } 
  delete delta;
  delete x;
  delete fi;
}



void
Beam::LUDcmp_lubksb(int *flag, double *da, double *b, double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder){
  int Ncol = 6;
  double tiny = 1.0e-20;

  for(int ie = 0; ie < Npart; ++ie){
    if(flag[ie] == 1)continue;//Skip the particle
    int imax;
    double vv[6];
    double big, dum, sum, temp;
    double a[6][6];
    int idx[6];

    for(int i = 0; i < Ncol; ++i){
      big = 0.0;
      for(int j = 0; j < Ncol; ++j){
	a[i][j] = da[Npart * (j * 6 + i) + ie];
	
	if((temp = fabs(a[i][j])) > big) big = temp;
      }
      if (big == 0){		    //The largest element is zero
	std::cout << "Singular matrix in routine LUDcmp" << ie << std::endl;
	exit(EXIT_FAILURE);
      }
      vv[i] = 1/big;		//Scaling
    }

    for (int j = 0; j < Ncol; ++j){	//Loop over columns of  Crout's method
      for (int i = 0; i < j; ++i){
	sum = a[i][j];
	for (int k = 0; k < i; ++k) sum -= a[i][k]*a[k][j];
	a[i][j] = sum;
      }
    
      //Search for largest pivot element
      big = 0.0;
      for (int i = j; i < Ncol; ++i){
	sum = a[i][j];
	for (int k = 0; k < j; ++k) sum -= a[i][k]*a[k][j];
	a[i][j] = sum;
	if ( (dum = vv[i]*fabs(sum))>=big){
	  big = dum;
	  imax = i;
	}
      }
      
      if (j != imax){			// Do we need to interchange rows?
	for (int k = 0; k < Ncol; ++k){	//Yes
	  dum = a[imax][k];
	  a[imax][k] = a[j][k];
	  a[j][k] = dum;
	}
	//d = -d;				//change the parity of d;
	vv[imax] = vv[j];	//Interchange the scale factor;
      }
		
      idx[j] = imax;
		
      if(a[j][j] == 0)	a[j][j] = tiny;	//If the pivot element is zero, submitted by a tiny value
		
      if(j!=(Ncol-1)){		      //Divide by the pivot element
	dum = 1.0/(a[j][j]);
	for (int i = j+1; i < Ncol; ++i) a[i][j] *= dum;
      }		
    }
    //Go back to the next column in the reduction

    
    //for(int i = 0; i < Ncol; ++i){
    //if(ie == 0)
    //	std::cout << "=>" << idx[i] << "\t" << xf[i * Npart + ie] << "\n";
    //}
    
    
    int ii=-1, ip;
  
    for(int i = 0; i< 6; ++i){
      ip = idx[i];
      sum = b[ip * Npart + ie];
      b[ip * Npart + ie] = b[i * Npart + ie];
      if(ii+1)
	for(int j=ii; j<=i-1; ++j)	
	  sum -= a[i][j] * b[j * Npart + ie];
      else if (sum) ii=i;
      b[i] = sum; 
    }
    
    for (int i = Ncol-1; i >= 0; --i){
      sum = b[i * Npart + ie];
      for(int j = i+1;j < Ncol; ++j)	{
	sum -= a[i][j] * b[j * Npart + ie];
      }
      b[i * Npart + ie] = sum/a[i][i];
    }
    
    for(int i=0; i< Ncol; ++i) xf[i * Npart + ie] -= b[i * Npart + ie];

    //for(int i = 0; i < Ncol; ++i){
    //if(ie == 0)
    //	std::cout << xf[i * Npart + ie] << "\n";
    //}
			
  }


}

int
Beam::checkResidual(double *f, int *flag, double epsabs, int Npart){
  int Ncol = 6;
  int dflag = 0;
  for(int ie = 0; ie < Npart; ++ie){
    double residual = 0.0;
    for(int i = 0; i < Ncol; ++i){
      residual += fabs(f[i * Npart + ie]);
    }	
    if(residual < epsabs) 
      flag[ie] = 1;
    else
      flag[ie] = 0;
    //std::cout << residual << "\t" << flag[ie] << "\n"; 
    dflag += flag[ie];
  }
  return dflag;
}

void
Beam::newtonIter(double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder){
  int Ncol = 6;
  double *f = new double[Npart * Ncol];
  gf2Eqns(f, M, ind, xi, xf, Nrow, maxLen, Npart, Norder);

  int nIter = 8;
  double *jac = new double[Npart * Ncol * Ncol];

  int *flag = new int[Npart];
  
  for(int i = 0; i < nIter; ++i){
    //TODO : performance checkResidual for every particle....
    int numValidParticles = checkResidual(f, flag, 1e-10, Npart);    
    //std::cout << i << " " << numValidParticles << std::endl;
    if(numValidParticles == Npart){
      delete f;
      delete jac;
      return ;
    }  
    Jacobian(flag, jac, M, ind, xi, xf, Nrow, maxLen, Npart, Norder);
    LUDcmp_lubksb(flag, jac, f, M, ind, xi, xf, Nrow, maxLen, Npart, Norder);
    gf2Eqns(f, M, ind, xi, xf, Nrow, maxLen, Npart, Norder);
  }
  delete f;
  delete jac;

}


void Beam::applyM(double *M, int *ind, double *x, int *Nrow, int maxLen, int Npart, int Norder){
  int Ncol = 6;
 
  double *xterm1 = new double[Npart];
  double *Mxterm1 = new double[Npart];

  double *xold = new double[Npart * Ncol];
  memcpy(xold, x, sizeof(double) * Npart * Ncol);
  std::cout << std::setprecision(16);
  
  for(int i = 1; i <= Ncol; ++i){
    for(int ie = 0; ie < Npart; ++ie)Mxterm1[ie] = 0;

    for(int j = 1; j <= Nrow[i - 1]; ++j){
      for(int ie = 0; ie < Npart; ++ie)xterm1[ie] = 1.0;

      for(int k = 1; k <= 6; ++k){
	for(int ie = 0; ie < Npart; ++ie){
	  xterm1[ie] = xterm1[ie] * pow(xold[ie + (k - 1) * Npart], ind[(j - 1) * Ncol * Ncol + (i - 1) * Ncol + k - 1]);
	  //std::cout << xterm1[ie] << "\t" << xold[ie * Ncol + k - 1] << "\t" << ind[(j - 1) * Ncol * Ncol + (i - 1) * Ncol + k - 1] << "\n";
	  
	}
      }
      for(int ie = 0; ie < Npart; ++ie){
	Mxterm1[ie] += M[(j - 1) * Ncol + (i - 1)]*xterm1[ie];
      } 
    }    
    for(int ie = 0; ie < Npart; ++ie){
      x[ie + (i - 1)*Npart] = Mxterm1[ie];
    }
  }  
}


double
Beam::slice(int *hOutOfBound, double *x, double *&x_s, int *&iSize, int Npart, int N, int Ncol, int *&id_s){
  Util::init(iSize, N, 0);
  double zmin  = Util::minval(x + 4 * Npart, Npart);    //! zmin (boundary) is the outlier
  double L     = 2.0 * fabs(zmin);   //! make the box size L = 2*|zmin|
  double dz    = L/double(N);
  
  for(int jk = 0; jk < Npart * Ncol * N; jk++)
  {
	id_s[jk] = -1;
  }
  
  for(int i = 0; i < Npart; ++i){

	if(hOutOfBound[i] == 1)
		continue;

    int iz = (x[4 * Npart + i] - zmin)/dz + 1;
    iz = MIN(MAX(1, iz), N);
    iz = iz - 1;
    iSize[iz] = iSize[iz] + 1;

   id_s[iz * Npart * Ncol + iSize[iz] - 1] = i;

    for(int j = 0; j < Ncol; ++j){
      x_s[iz * Npart * Ncol + j * Npart + iSize[iz] - 1] = x[j * Npart + i];
    }
  }
  
/*  for(int k=0; k < Npart * (Ncol + 2); k++)
  {
	std::cout<<"k is : "<<k<<" Main Array is : "<<x[k]<<"\n";
  }
  for(int kk=0; kk < Npart * Ncol * N; kk++)
  {
	std::cout<<"kk is : "<<kk<<" Slice Array is : "<<x_s[kk]<<"\n";
  }
  for(int kj=0; kj < Npart * Ncol * N; kj++)
  {
	std::cout<<"kj is : "<<kj<<" Index Array is : "<<id_s[kj]<<"\n";
  } */
  return zmin;
}
/*
void
Beam::mergeSlices(double *&x, double *x_s, int *iSize, int Npart, int N, int Ncol){
  int i = 0;
  for(int iSlice = 0; iSlice < N; ++iSlice){
    for(int iPart = 0; iPart < iSize[iSlice]; ++iPart){
      for(int j = 0; j < Ncol; ++j){
	x[j * Npart + i] = x_s[iSlice * Npart * Ncol + j * Npart + iPart];
      }
      ++i;
    }
  }
}
*/

void
Beam::mergeSlices(double *&x, double *x_s, int *iSize, int Npart, int N, int Ncol, int *id_s){

int index = 0;

for(int kj=0; kj < Npart * Ncol * N; kj++)
  {
        if(id_s[kj] != -1)
		{
			index = id_s[kj];
			x[index] = x_s[kj];
                        x[index + Npart] = x_s[kj + Npart];
                        x[index + Npart * 2] = x_s[kj + Npart * 2];
                        x[index + Npart * 3] = x_s[kj + Npart * 3];
                        x[index + Npart * 4] = x_s[kj + Npart * 4];
                        x[index + Npart * 5] = x_s[kj + Npart * 5];		
		
		}
  }
/*
	std::cout<<"------------------------------------------------------------------------------------------- \n";
	std::cout<<"------------------------------------------------------------------------------------------- \n";
  for(int k=0; k < Npart * Ncol; k++)
  {
        std::cout<<"k is : "<<k<<" Main Array after merge is : "<<x[k]<<"\n";
  }
*/
}

void
Beam::collide(double *x_ebunch, double *x_pbunch, int *hOutOfBound_e, int *hOutOfBound_p, BeamParams *bParams, int Ncol){
  int N = bParams->N;
  double *x_es = new double[bParams->Npart_e * Ncol * N];
  double *x_ps = new double[bParams->Npart_p * Ncol * N];
  int *iSize_e = new int[N];
  int *iSize_p = new int[N];

  int *id_e = new int[bParams->Npart_e * Ncol * N];
  int *id_p = new int[bParams->Npart_p * Ncol * N];

  for(int k=0; k < bParams->Npart_e * Ncol * N; k++)
  {
	x_es[k] = 0;
	x_ps[k] = 0;
  }

  /*
  for(int k=0; k < bParams->Npart_e * Ncol; k++)
  {
        std::cout<<"k is : "<<k<<" Original Main Array is : "<<x_e[k]<<"\n";
  }

  for(int k=0; k < bParams->Npart_p * Ncol; k++)
  {
        std::cout<<"k is : "<<k<<" Original Main Array is : "<<x_p[k]<<"\n";
  }
*/
  double zmin_e = slice(hOutOfBound_e, x_ebunch, x_es, iSize_e, bParams->Npart_e, N, Ncol, id_e);
  double zmin_p = slice(hOutOfBound_p, x_pbunch, x_ps, iSize_p, bParams->Npart_p, N, Ncol, id_p);
  double L_e     = 2.0 * fabs(zmin_e); 
  double L_p     = 2.0 * fabs(zmin_p);

	std::cout<<"zmin_e is : "<<zmin_e<<"\n";
	std::cout<<"zmin_p is : "<<zmin_p<<"\n";


  
  //!-------------------------------------------------
  //! Set the drift length for multi-slice collision. 
  //!-------------------------------------------------
  double delta_e = 0.0;
  double delta_p = 0.0;
  if (N != 1) {
    delta_e = L_e/double(N);
    delta_p = L_p/double(N);
  }

  double *z_e = new double[N];
  double *z_p = new double[N];
  if (N == 1) {
    z_e[0] = 0.0;
    z_p[0] = 0.0;
  }else{
    int j = 0;
    for(int i = -(N - 1); i <= (N - 1); i+=2){
      j = j + 1;
      z_e[j - 1] = -i * delta_e/2.0;
      z_p[j - 1] = -i * delta_p/2.0;
    }
  }


  //!-------------------------------------------------------
  //! Collide appropriate slices and apply drifts by drift
  //!-------------------------------------------------------
  //!   Example: N = 5
  //!--
  //! T  2:                                  11
  //! O  3:                          12              21
  //! P  4:                  13              22              31
  //!    5:          14              23              32              41
  //!__  6:  15              24              33              42              51
  //! B  7:          25              34              43              52
  //! O  8:                  35              44              53
  //! T  9:                          45              54
  //!   10:                                  55
  //!-----------------------------------------------------------------
  //! Top triangle
  //!---------------


  for(int i = 2; i <= N+1; ++i){
    for(int j = 1; j <= i-1; ++j){
      double S = (z_p[i-j-1]-z_e[j-1])/2.0;
      applyKick(bParams, &x_es[(j - 1) * bParams->Npart_e * Ncol], &x_ps[(i - j - 1) * bParams->Npart_p * Ncol], iSize_e[j-1], iSize_p[i-j-1], S, j-1, i-j-1);            
    }
  }

  //!-----------------
  //! Bottom triangle
  //!-----------------

  for(int i = N + 2; i <= 2*N; ++i){
    for(int j = i - N; j<= N; ++j){
      double S = (z_p[i-j-1]-z_e[j - 1])/2.0;
      applyKick(bParams, &x_es[(j - 1) * bParams->Npart_e * Ncol], &x_ps[(i - j - 1) * bParams->Npart_p * Ncol], iSize_e[j-1], iSize_p[i-j-1], S, j-1, i-j-1);                  
    }
  }


  mergeSlices(x_ebunch, x_es, iSize_e, bParams->Npart_e, N, Ncol, id_e);
  mergeSlices(x_pbunch, x_ps, iSize_p, bParams->Npart_p, N, Ncol, id_p);

}


void 
Beam::applyKick(BeamParams *bParams, double *x_es, double *x_ps, int iSe, int iSp, double S, int iE, int iP){
  double r_e = 2.817940289458e-15;
  double r_p = 1.534698e-18;
  std::complex<double> eye (0.0,1.0), z1, z2, w1, w2, fk;
  double Fx, Fy;

  //!--------------------
  //! STRONG-STRONG MODE
  //!------------------------------------------------------------------------
  //! Compute kicks according to the Handbook of Accelerator Physics,
  //! Section 2.6.1.1. Eqs.(4)-(5) and Section 2.6.1.2. Eqs.(32)-(35),
  //! generalized to strong-strong (weak-weak) regime (AH has weak-strong).
  //!------------------------------------------------------------------------
  if(bParams->iRegime == 2){
    //!----------------------------------------------------------------------------
    //! Compute the mean and the standard deviation of the particle distribution 
    //! in each of the two transversal coordinates for each beam 
    //!----------------------------------------------------------------------------
    double xbar_e = 0, ybar_e = 0;
    double sig_x_e = 0, sig_y_e = 0, sg_e = 0;

    double sxy_e = 0;
    double syx_e = 0;

    double sxy_p = 0;
    double syx_p = 0;


    if(iSe > 1){

      for(int i = 0; i < iSe; ++i){
	xbar_e += (x_es[i] - S * x_es[1 * bParams->Npart_e + i]);
	ybar_e += (x_es[2 * bParams->Npart_e + i] - S * x_es[3 * bParams->Npart_e + i]);
      }
	

      xbar_e = xbar_e/double(iSe);
      ybar_e = ybar_e/double(iSe);

      for(int i = 0; i < iSe; ++i){
	sig_x_e += pow((x_es[i] - S * x_es[1 * bParams->Npart_e + i] - xbar_e), 2.0);
	sig_y_e += pow((x_es[2 * bParams->Npart_e + i] - S * x_es[3 * bParams->Npart_e + i] - ybar_e), 2.0);
      }      
      sig_x_e = sqrt(sig_x_e/double(iSe));
      sig_y_e = sqrt(sig_y_e/double(iSe));
      sg_e = 1.0/sqrt(2.0*(fabs(sig_x_e*sig_x_e - sig_y_e*sig_y_e)));

      if(sig_x_e != 0 && sig_y_e != 0)
      {
      sxy_e = sig_x_e/sig_y_e;
      syx_e = sig_y_e/sig_x_e;
      }

    }

    double xbar_p = 0, ybar_p = 0;
    double sig_x_p = 0, sig_y_p = 0, sg_p = 0;

    if(iSp > 1){
      for(int i = 0; i < iSp; ++i){
	xbar_p += (x_ps[i] + S * x_ps[1 * bParams->Npart_p + i]);
	ybar_p += (x_ps[2 * bParams->Npart_p + i] + S * x_ps[3 * bParams->Npart_p + i]);
      }
      xbar_p = xbar_p/double(iSp);
      ybar_p = ybar_p/double(iSp);

      for(int i = 0; i < iSp; ++i){
	sig_x_p += pow((x_ps[i] + S * x_ps[1 * bParams->Npart_p + i] - xbar_p), 2.0);
	sig_y_p += pow((x_ps[2 * bParams->Npart_p + i] + S * x_ps[3 * bParams->Npart_p + i] - ybar_p), 2.0);
      }      
      sig_x_p = sqrt(sig_x_p/double(iSp));
      sig_y_p = sqrt(sig_y_p/double(iSp));
      sg_p = 1.0/sqrt(2.0*(fabs(sig_x_p*sig_x_p - sig_y_p*sig_y_p)));

      if(sig_x_p != 0 && sig_y_p != 0)
      {
      sxy_p = sig_x_p/sig_y_p;
      syx_p = sig_y_p/sig_x_p;
      }

    }

    double fcnst_e = (iSp*r_e/bParams->gamma_e)*(2.0*sqrt(PI)*sg_p)*(bParams->N_p/bParams->Npart_inbound_p);
    double fcnst_p = (iSe*r_p/bParams->gamma_p)*(2.0*sqrt(PI)*sg_e)*(bParams->N_e/bParams->Npart_inbound_e);

    std::cout << "--------------\n";
    std::cout << "slice\t" << iE << "\n";
    std::cout << "S\t" << S << "\n";
    std::cout << "eslice\t" << iSe << "\n";
    std::cout << "pslice\t" << iSp << "\n";
    std::cout << "xbar_e\t" << xbar_e << "\n";
    std::cout << "ybar_e\t" << ybar_e << "\n";
    std::cout << "xbar_p\t" << xbar_p << "\n";
    std::cout << "ybar_p\t" << ybar_p << "\n";
    std::cout << "sig_x_e\t" << sig_x_e << "\n";
    std::cout << "sig_y_e\t" << sig_y_e << "\n";
    std::cout << "sig_x_p\t" << sig_x_p << "\n";
    std::cout << "sig_y_p\t" << sig_y_p << "\n";
    std::cout << "sxy_e\t" << sxy_e << "\n";
    std::cout << "syx_e\t" << syx_e << "\n";
    std::cout << "sxy_p\t" << sxy_p << "\n";
    std::cout << "syx_p\t" << syx_p << "\n";
    
    //!---------------------------------------------------
    //! Apply beam-beam kicks to particles in the e-beam
    //!---------------------------------------------------

    for(int i = 0; i < iSe; ++i){
      double x1 = x_es[i] - S * x_es[1 * bParams->Npart_e + i] - xbar_p;
      double y1 = x_es[2 * bParams->Npart_e + i] - S * x_es[3 * bParams->Npart_e + i] - ybar_p;
      double eterm = exp(-0.50 * pow((x1/sig_x_p), 2.0) -0.50 * pow((y1/sig_y_p), 2.0));
      x1 = x1*sg_p;
      y1 = y1*sg_p;
      double x2 = syx_p*x1;
      double y2 = sxy_p*y1;

      if (sig_x_p > sig_y_p) {
	z1 = x1 + eye*fabs(y1);    //! z1 = x1 + eye*y1
	z2 = x2 + eye*fabs(y2);    //! z2 = x2 + eye*y2
//	std::cout<<"z1 is : "<<z1<<"\n";
	w1 = Util::WOFZ(std::real(z1), std::imag(z1));
	w2 = Util::WOFZ(std::real(z2), std::imag(z2));

	if (fcnst_e == 0.0) {
	  fk = std::complex<double>(0.0, 0.0);
	}else{
	  fk = fcnst_e*(w1 - eterm*w2);
	}

	if (y1 > 0.0) {
	  Fx = std::imag(fk);
	  Fy = std::real(fk);
	}else{
	  Fx = std::imag(fk);
	  Fy =-std::real(fk);           
	}      
      }else{
	z1 = y1 + eye*fabs(x1);    
	z2 = y2 + eye*fabs(x2);   
  //      std::cout<<"z1 is : "<<z1<<"\n"; 
	w1 = Util::WOFZ(std::real(z1), std::imag(z1));
	w2 = Util::WOFZ(std::real(z2), std::imag(z2));

	//fk = fcnst_p*(w1 - eterm*w2);
         fk = fcnst_e*(w1 - eterm*w2);

	if(x1 < 0.0){
	  Fx =-std::real(fk);
	  Fy = std::imag(fk);
	}else{
	  Fx = std::real(fk);
	  Fy = std::imag(fk);
	}
      }
	
	//std::cout<<"Fx is : "<<Fx<<"\n";
      x_es[0 * bParams->Npart_e + i] -= S * Fx;
      x_es[1 * bParams->Npart_e + i] -= Fx;
      x_es[2 * bParams->Npart_e + i] -= S * Fy;
      x_es[3 * bParams->Npart_e + i] -= Fy;
    }

    for(int i = 0; i < iSp; ++i){
      double x1 = x_ps[i] + S * x_ps[1 * bParams->Npart_p + i] - xbar_e;
      double y1 = x_ps[2 * bParams->Npart_p + i] + S * x_ps[3 * bParams->Npart_p + i] - ybar_e;
      double eterm = exp(-0.50 * pow((x1/sig_x_e), 2.0) -0.50 * pow((y1/sig_y_e), 2.0));
      x1 = x1*sg_e;
      y1 = y1*sg_e;
      double x2 = syx_e*x1;
      double y2 = sxy_e*y1;

      if (sig_x_e > sig_y_e) {
	z1 = x1 + eye*fabs(y1);    //! z1 = x1 + eye*y1
	z2 = x2 + eye*fabs(y2);    //! z2 = x2 + eye*y2
    //    std::cout<<"z1 is : "<<z1<<"\n";
	w1 = Util::WOFZ(std::real(z1), std::imag(z1));
	w2 = Util::WOFZ(std::real(z2), std::imag(z2));


	if (fcnst_p == 0.0) {
	  fk = std::complex<double>(0.0, 0.0);
	}else{
	  fk = fcnst_p*(w1 - eterm*w2);
	}

	//std::cout << fk << "\t" << fcnst_p << "\n";

	if (y1 > 0.0) {
	  Fx = std::imag(fk);
	  Fy = std::real(fk);
	}else{
	  Fx = std::imag(fk);
	  Fy =-std::real(fk);           
	}      
	//std::cout << Fx << "\t" << Fy << "\n";

      }else{
	z1 = y1 + eye*fabs(x1);    
	z2 = y2 + eye*fabs(x2);    
      //  std::cout<<"z1 is : "<<z1<<"\n";
	w1 = Util::WOFZ(std::real(z1), std::imag(z1));
	w2 = Util::WOFZ(std::real(z2), std::imag(z2));

	fk = fcnst_p*(w1 - eterm*w2);
	if(x1 < 0.0){
	  Fx =-std::real(fk);
	  Fy = std::imag(fk);
	}
	Fx = std::real(fk);
	Fy = std::imag(fk);
	
      }

      //std::cout << x_ps[0 * bParams->Npart_p + i] << "\t" << x_ps[1 * bParams->Npart_p + i] << "\t" << x_ps[2 * bParams->Npart_p + i] << "\t" << x_ps[3 * bParams->Npart_p + i] << "\n";
      //std::cout << S << "\t" << Fx << "\t" << Fy << "\n";
        //std::cout<<"Fx is : "<<Fx<<"\n";

      x_ps[0 * bParams->Npart_p + i] += S * Fx;
      x_ps[1 * bParams->Npart_p + i] -=  Fx;
      x_ps[2 * bParams->Npart_p + i] += S * Fy;
      x_ps[3 * bParams->Npart_p + i] -=  Fy;

	//std::cout<<"x_ps is : "<<x_ps[0 * bParams->Npart_p + i]<<"\n";

    } 
  }
}

