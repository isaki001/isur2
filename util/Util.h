#ifndef UTIL_H
#define UTIL_H

#define MAX(a, b) ((a > b)?a:b)
#define MIN(a, b) ((a < b)?a:b)

#include <complex>
class Util{
 public:
  static std::complex<double> WOFZ(double, double);
  static int cIDNINT(double );

  static double gauss(double xbar, double sigma, double sigma_max, int &Nseed);
  static double ran1(int &);
  static double gasdev(int &);
  
  template<typename T>
    static void
    init(T *data, int size, T val){
    for(int i = 0; i < size; ++i){
        data[i] = val;
    }
  }

  template<typename T>
    static T
    minval(T *data, int size){
    T min = data[0];
    for(int i = 1; i < size; ++i){
      if( data[i] < min)
	min = data[i];
    }
    return min;
  }

  template<typename T>
    static T
    maxval(T *data, int size){
    T max = data[0];
    for(int i = 1; i < size; ++i){
      if( data[i] > max)
	max = data[i];
    }
    return max;
  }

};

#endif /* UTIL_H */
