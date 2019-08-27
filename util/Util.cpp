#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>

#include "Util.h"



int iy = 0;
int iv[32]={0};
int iset = 0;
double gset = 0;
  



double
Util::ran1(int &idum){
  const int ia = 16807;
  const long int im = 2147483647;
  double am = 1.0/im;
  const int iq = 127773, ir = 2836;
  const int ntab = 32;
  const int ndiv = 1 + (im-1)/ntab;
  const double eps=1.2e-7;
  double rnmx=1.0 - eps;
  
  int k = 0, j = 0;
  if(idum <= 0 || iy == 0){
    idum = MAX(-idum,1);
    for(j = ntab+8; j >=1; j--){
      k = idum/iq;
      idum = ia * (idum - k*iq) - ir*k;
      if (idum < 0) idum = idum + im;
      if (j <= ntab) iv[j - 1] = idum;
    }
    iy = iv[0];
  }
  k = idum/iq;
  idum = ia * (idum - k*iq) - ir * k;
  if (idum < 0) idum = idum + im;
  j = 1 + iy/ndiv;
  iy = iv[j - 1];
  iv[j - 1] = idum;
  double retval= MIN((am*iy), rnmx);
  return retval;
}


double
Util::gasdev(int &idum){
  double retval = 0;
  if(iset == 0){
    double rsq = 0, v1 = 0, v2 = 0;
    do{
      v1 = 2.0 * ran1(idum) - 1.0;
//	std::cout<<"v1 is : "<<v1<<"\n";
      v2 = 2.0 * ran1(idum) - 1.0;
      rsq = v1*v1 + v2*v2;
    }while ((rsq >= 1.0) || (rsq == 0.0)); 
    double fac = sqrt(-2.0 * log(rsq)/rsq);
    gset = v1 * fac;
    retval = v2 * fac;
    iset = 1;
  }else{
    retval = gset;
    iset = 0;
  }
  return retval;
}


double
Util::gauss(double xbar, double sigma, double sigma_max, int &Nseed){
  double retval = sigma*gasdev(Nseed);
  return retval;
}


std::complex<double> 
Util::WOFZ(double XI, double YI){
  double Ureal=0.0,Vimg=0.0;
  bool A,B;

  double H=0.0,H2=0.0,U2=0.0,V2=0.0;
  int KAPN=0,NU=0;

  double FACTOR   = 1.12837916709551257388,
    RMAXREAL = 0.5e+154,
    RMAXEXP  = 708.503061461606,
    RMAXGONI = 3.53711887601422e+15;

  double XABS = fabs(XI);
  double YABS = fabs(YI);
  double X    = XABS/6.3;
  double Y    = YABS/4.4;

  //printf("%d %.16e %.16e\n", blockIdx.x * blockDim.x + threadIdx.x, XABS, YABS);
  
  if ((XABS > RMAXREAL) || (YABS > RMAXREAL)) {
    std::complex<double> val (Ureal, Vimg);
    return val;
  }

  double QRHO = X*X + Y*Y;

  double XABSQ = XABS*XABS;
  double XQUAD = XABSQ - YABS*YABS;
  double YQUAD = 2*XABS*YABS;

  A = (QRHO < ((double)0.085264));


  if(A) {
      QRHO  = (1-0.85 * Y ) * sqrt(QRHO);
      int N = cIDNINT(6 + 72 * QRHO);
      int J = 2 * N + 1;
      double XSUM  = 1.0/J;
      double YSUM  = 0.0;

      int i;
      for (i=N;i>=1;i--){
	J = J - 2;
	double XAUX = (XSUM * XQUAD - YSUM*YQUAD)/i;
	YSUM = (XSUM*YQUAD + YSUM*XQUAD)/i;
	XSUM = XAUX + 1.0/J;
      }
      double U1   = -FACTOR*(XSUM*YABS + YSUM*XABS) + 1.0;
      double V1   =  FACTOR*(XSUM*XABS - YSUM*YABS);
      
      double DAUX =  exp(-XQUAD);
      U2   =  DAUX*cos(YQUAD);
      V2   = -DAUX*sin(YQUAD);
      
      Ureal    = U1*U2 - V1*V2;
      Vimg    =  U1*V2 + V1*U2;   
  }else{
    if (QRHO > 1.0){
      H = 0.0;
      KAPN = 0;
      QRHO = sqrt(QRHO);
      NU   = cIDNINT(3 + (1442/(26*QRHO+77)));
    }else{
      QRHO = (1-Y)*sqrt(1-QRHO);
      H    = 1.88*QRHO;
      H2   = 2*H;
      KAPN = cIDNINT(7  + 34 * QRHO);
      NU   = cIDNINT(16 + 26 * QRHO);
    }
    B = (H > 0.0);
    double QLAMBDA = 0.0;

    if (B) QLAMBDA = pow(H2,KAPN);

    double RX = 0.0;
    double RY = 0.0;
    double SX = 0.0;
    double SY = 0.0;

    int N2;
    for(N2=NU;N2>=0;N2--){
      int NP1 = N2 + 1;
      double TX  = YABS + H + NP1 * RX;
      double TY  = XABS - NP1*RY;
      double C   = 0.5/(TX*TX + TY*TY);
      RX  = C*TX;
      RY  = C*TY;
      if ((B)&&(N2<=KAPN)){
	TX = QLAMBDA + SX;
	SX = RX*TX - RY*SY;
	SY = RY*TX + RX*SY;
	QLAMBDA = QLAMBDA/H2;
      }
    }  
    if(H == 0.0){
      Ureal = FACTOR*RX;
      Vimg = FACTOR*RY;
    }else{
      Ureal = FACTOR*SX;
      Vimg = FACTOR*SY;
    }  
    if (YABS == 0.0) 
      Ureal = exp(-((XABS)*(XABS))); /* Not Sure */
  }

  if (YI < 0.0){	
    if (A){
      U2    = 2*U2;
      V2    = 2*V2;
    }else{
      XQUAD =  -XQUAD;
      if ((YQUAD > RMAXGONI)||(XQUAD > RMAXEXP)){
	std::complex<double> val (Ureal, Vimg);
	return val;
      }
      double W1 =  2  * exp(XQUAD);
      U2 =  W1 * cos(YQUAD);
      V2 = -(W1 * sin(YQUAD));
    }
    Ureal = U2 - (Ureal);
    Vimg = V2 - (Vimg);
    if (XI > 0.0) 
      Vimg = -(Vimg);
  }else{
    if (XI < 0.0) 
      Vimg = -(Vimg);
  }

  std::complex<double> val (Ureal, Vimg);
  return val;
} 

int 
Util::cIDNINT(double rval){
  if(rval < 0.0)
    return (rval - 0.5);
  else
    return (rval + 0.5);
}
