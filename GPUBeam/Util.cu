


/**
   reference:
   http://www.yolinux.com/TUTORIALS/LinuxTutorialMixingFortranAndC.html
*/	
__device__ int cIDNINT(double rval)
{
  if(rval < 0.0)
    return (rval - 0.5);
  else
    return (rval + 0.5);
}


__device__ cuDoubleComplex WOFZ (double XI, double YI)
{
  
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
    return make_cuDoubleComplex(Ureal, Vimg);
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
      int i;//, ii = 0;

      for (i=N;i>=1;i--){
	//if (ii == 3) break;
	J = J - 2;
	double XAUX = (XSUM * XQUAD - YSUM*YQUAD)/i;
	YSUM = (XSUM*YQUAD + YSUM*XQUAD)/i;
	XSUM = XAUX + 1.0/J;
	//ii++;
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

    if (B) QLAMBDA = pow(H2,(double)KAPN);

    double RX = 0.0;
    double RY = 0.0;
    double SX = 0.0;
    double SY = 0.0;

    int N2;//, N22 = 0;

    for(N2=NU;N2>=0;N2--){
	//if (N22 == 3) break;
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
	//N22++;
    }  
    if(H == 0.0){
      Ureal = FACTOR*RX;
      Vimg = FACTOR*RY;
    }else{
      Ureal = FACTOR*SX;
      Vimg = FACTOR*SY;
    }  
    if (YABS == 0.0) 
      Ureal = exp(-((XABS)*(XABS))); //Not Sure
  }

  if (YI < 0.0){	
    if (A){
      U2    = 2*U2;
      V2    = 2*V2;
    }else{
      XQUAD =  -XQUAD;
      if ((YQUAD > RMAXGONI)||(XQUAD > RMAXEXP)) 
	return make_cuDoubleComplex(Ureal, Vimg); 
     
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
  return make_cuDoubleComplex(Ureal, Vimg); 


} 


__device__ cuDoubleComplex zwofz(double X1, double Y1)
{
  /* Local variables */
  double rsqpd2 = 1.128379167095512573896158903121545171688;
  double mxgoni = 59473682.;
  double inogxm = 1.6814159916986475e-8;
  double lgovfl  = 708.;
  int kapn, flag, itmp;
  double xabs, yabs, qrho, xsum, ysum, dtmp;
  int fadeeva;
  double c, h;
  int i, j, n;
  double u, v, x, y, xquad, yquad, h2, u1, u2, v1, v2, w1, qlamda;
  int nu;
  double rx, ry, sx, sy, tx, ty, uv;
  double Ureal=0.0,Vimg=0.0;

  /* >>1992-10-13 ZWOFZ WVS Improve efficiency and avoid underflow. */
  /* >>1992-03-13 ZWOFZ FTK Removed implicit statements. */
  /* >>1991-08-23 ZWOFZ WV Snyder Initial adaptation to Math77 */
  /* >> HMP modified for IEEE parameters and no error print */
  /* >>> note that flag = 0 for yi > 0 */

  /*     Algorithm 680, collected algorithms from ACM. */
  /*     This work published in Transactions on Mathematical Software, */
  /*     Vol. 16, No. 1, Pp. 47. */

  /*     Modified by W. V. Snyder for inclusion in Math77: */
  /*     Reorganize checking for overflow and loss of precision so */
  /*     there are no redundant or unnecessary checks.  In the process, */
  /*     the range of applicability is expanded to the entire upper */
  /*     half-plane. */
  /*     Reorganize some calculations to be immune from overflow. */
  /*     Split loop for two outer regions into two loops -- faster in */
  /*     region. */
  /*     Use D1MACH to fetch machine characteristics. (NOT) */
  /*     Use Math77 error message processor. */

  /*  Given a complex number z = (xi,yi), this subroutine computes */
  /*  the value of the Faddeeva function w(z) = exp(-z**2)*erfc(-i*z), */
  /*  where erfc is the complex complementary error function and i */
  /*  means sqrt(-1). */
  /*  The accuracy of the algorithm for z in the 1st and 2nd quadrant */
  /*  is 14 significant digits; in the 3rd and 4th it is 13 significant */
  /*  digits outside a circular region with radius 0.126 around a zero */
  /*  of the function. */


  /*  Argument list */
  /*     Z [in]    = real and imaginary parts of z in Z(1) and Z(2) */
  /*     W [out]   = real and imaginary parts of w(z) in W(1) and W(2) */
  /*     FLAG [out] = an error flag indicating the status of the */
  /*       computation.  Type INTEGER, with values having the following */
  /*       meaning: */
  /*         0 : No error condition, */
  /*        -1 : Overflow would occur, */
  /*        +1 : There would be no significant digits in the answer. */


  /*  The routine is not underflow-protected but any variable can be */
  /*  put to zero upon underflow. */

  /*  Reference - GMP Poppe, CMJ Wijers: More efficient computation of */
  /*  the complex error-function, ACM Trans. Math. Software. */



  /*     RSQPD2 = 2/sqrt(pi) */
  /*     SQLGOV = sqrt(ln(RMAX)), where RMAX = the overflow limit for */
  /*              floating point arithmetic. */
  /*     LGOVFL  = ln(RMAX) - ln(2) */
  /*     LGUND  = ln(RMIN), where RMIN = underflow limit. */
  /*     MXGONI = the largest possible argument of sin or cos, restricted */
  /*              here to sqrt ( pi / (2*round-off-limit) ). */
  /*     INOGXM = 1 / MXGONI. */
  /*  The reason these values are needed as defined */
  /*  will be explained by comments in the code. */

  xabs = fabs(X1);
  yabs = fabs(Y1);
  x = xabs / 6.3;
  y = yabs / 4.4;
  
  if (x > y) {
    dtmp = y / x;
    qrho = x * sqrt(dtmp * dtmp + 1.);
  } else if (y == 0.) {
    qrho = 0.;
  } else {
    dtmp = x / y;
    qrho = y * sqrt(dtmp * dtmp + 1.);
  }

  fadeeva = (qrho < 0.292);
  if (fadeeva) {

    /*  qrho .lt. 0.292, equivalently qrho**2 .lt. 0.085264: the Fadeeva */
    /*  function is evaluated using a power-series (Abramowitz and */
    /*  Stegun, equation (7.1.5), p.297). */
    /*  N is the minimum number of terms needed to obtain the required */
    /*  accuracy. */
    /*  We know xquad and exp(-xquad) and yqyad and sin(yquad) won't */
    /*  cause any trouble here, because qrho .lt. 1. */
    xquad = (xabs - yabs) * (xabs + yabs);
    yquad = 2. * xabs * yabs;
    n = (int) ((1. - 0.85 * y) * 72. * qrho + 6.5);
    j = (n << 1) + 1;
    xsum = rsqpd2 / j;
    ysum = 0.;
    for (i = n; i > 0; --i) {
      j -= 2;
      w1 = (xsum * xquad - ysum * yquad) / i;
      ysum = (xsum * yquad + ysum * xquad) / i;
      xsum = w1 + rsqpd2 / j;
    }
    u1 = 1. - (xsum * yabs + ysum * xabs);
    v1 = xsum * xabs - ysum * yabs;
    w1 = exp(-xquad);
    u2 = w1 * cos(yquad);
    v2 = -w1 * sin(yquad);
    u = u1 * u2 - v1 * v2;
    v = u1 * v2 + v1 * u2;
  } else {
    rx = 0.;
    ry = 0.;
    sx = 0.;
    sy = 0.;

    /*  The loops in both branches of the IF block below are similar. */
    /*  They could be combined to reduce space, but extra tests and */
    /*  unnecessary computation would be needed. */

    if (qrho < 1.) {
      /*  0.292 < qrho < 1.0: w(z) is evaluated by a truncated */
      /*  Taylor expansion, where the Laplace continued fraction */
      /*  is used to calculate the derivatives of w(z). */
      /*  KAPN is the minimum number of terms in the Taylor expansion */
      /*       needed to obtain the required accuracy. */
      /*  NU is the minimum number of terms of the continued fraction */
      /*     needed to calculate the derivatives with the required */
      /*     accuracy. */
      /*  x*x + y*y is more accurate than qrho*qrho here: */
      c = (1. - y) * sqrt(1. - x * x - y * y);
      h = 1.88 * c;  h2 = 2. * h;
      nu = (int) (26. * c + 17.5);
      kapn = (int) (34. * c + 8.5);
      /*  Select kapn so qlamda doesn't underflow.  Small kapn is good */
      /*         (when possible) for performance also. */
      if (h2 < 0.25) {
        itmp = (int) (-lgovfl / log(h2)) + 1;
        if (itmp < kapn) kapn = itmp;
      }
      qlamda = pow(h2, (double) (kapn - 1));
      /*  0 < qlamda < 3.76**41 < 3.85d23. */
      for (n = nu; n > kapn; --n) {
        tx = yabs + h + n * rx;
        ty = xabs - n * ry;
        /* No overflow because tx*rx + ty*ry = 1 and 0.292 < qrho < 1: */
        c = 0.5 / (tx * tx + ty * ty);
        rx = tx * c;
        ry = ty * c;
      }
      for (n = kapn; n > 0; --n) {
        tx = yabs + h + n * rx;
        ty = xabs - n * ry;
        /*  No overflow because tx*rx + ty*ry = 1 and 0.292 < qrho < 1: */
        c = 0.5 / (tx * tx + ty * ty);
        rx = tx * c;
        ry = ty * c;
        tx = qlamda + sx;
        sx = rx * tx - ry * sy;
        sy = ry * tx + rx * sy;
        qlamda /= h2;
      }
      u = sx * rsqpd2;
      v = sy * rsqpd2;
    } else {
      /*   qrho >= 1.: w(z) is evaluated using the Laplace continued */
      /*   fraction. */
      /*   NU is the minimum number of terms needed to obtain the */
      /*       required accuracy. */
      nu = (int) (1442. / (26. * qrho + 77.) + 4.5);
      for (n = nu; n > 0; --n) {
        tx = yabs + n * rx;
        ty = xabs - n * ry;
        if (tx > fabs(ty)) break;
        /*  rx = 0.5*tx/(tx**2+ty**2) and ry = 0.5*ty/(tx**2+ty**2), */
        /*  computed without overflow.  Underflow is OK. */
        c = tx / ty;
        ry = 0.5 / (ty * (c * c + 1.));
        rx = ry * c;
      }
      while (n > 0){
        /*  Once tx>abs(ty), it stays that way. */
        /*  rx = 0.5*tx/(tx**2+ty**2) and ry = 0.5*ty/(tx**2+ty**2), */
        /*  computed without overflow.  Underflow is OK. */
        c = ty / tx;
        rx = 0.5 / (tx * (c * c + 1.));
        ry = rx * c;
        if (--n == 0) break;
        tx = yabs + n * rx;
        ty = xabs - n * ry;
      }
      u = rx * rsqpd2;
      v = ry * rsqpd2;
    }
    if (yabs == 0.) {
      dtmp = xabs * xabs;
      if (dtmp > lgovfl) {
        u = 0.;
      } else {
        u = exp(-dtmp);
      }
    }
  }

  /*     Evaluation of w(z) in the other quadrants. */
  flag = 0;
  if (Y1 < 0.) { /* y is negative */
    if (fadeeva) {
      u = 2. * u2 - u;
      v = 2. * v2 - v;
    } else {
      /*  Check whether sin(2*xabs*yabs) has any precision, without */
      /*  allowing 2*xabs*yabs to overflow. */
      if (yabs > xabs) {
        if (yabs > inogxm) {
          /*  The following protects 2*exp(-z**2) against overflow. */
          if (lgovfl / yabs < yabs - xabs * (xabs / yabs)) 
            flag = -1;
        }else{
          w1 = 2. * exp((yabs - xabs) * (yabs + xabs));
          uv = fabs(u); dtmp = fabs(v);
          if (dtmp < uv) uv = dtmp; /* uv = min (abs(u), abs(v)) */
          dtmp = xabs * yabs;
          if (w1 < uv) dtmp *= (w1 / uv);
          if (dtmp > mxgoni) flag = 1;
        }
      } else if (xabs > inogxm) {
        if (lgovfl / xabs < xabs - yabs * (yabs / xabs)) {
          /*   (yabs-xabs)*(yabs+xabs) might have overflowed, but in that */
          /*   case, exp((yabs-xabs)*(yabs+xabs)) would underflow. */
          u = -u; v = -v; flag = 2;
        }else{
          /* (yabs-xabs)*(yabs+xabs) can't overflow here. */
          w1 = 2. * exp((yabs - xabs) * (yabs + xabs));
          uv = fabs(u); dtmp = fabs(v);
        }
	if (abs(u) < abs(v)){ uv = abs(u);} else {uv = abs(v);}
        if (dtmp < uv) uv = dtmp; /* uv = min (abs(u), abs(v)) */
        dtmp = xabs * yabs;
        if (w1 < uv) dtmp *= (w1 / uv);
        if (dtmp > mxgoni) flag = 1;
      }
    }
    if (flag == 0){
      yquad = 2. * xabs * yabs;
      u =  w1 * cos(yquad) - u;
      v = -w1 * sin(yquad) - v;
    }
    if (flag == 2) flag = 0;
  }
  if (flag) {
    Ureal = exp(lgovfl);
    Vimg = Ureal;
  }else{
    Ureal = u;
    if (X1 < 0.)  v = -v; 
    Vimg = v;
  }
  return make_cuDoubleComplex(Ureal, Vimg);
} /* zwofz */
