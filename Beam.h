#ifndef BEAM_H
#define BEAM_H

struct Map{
  int *Nrow;
  double *M;
  int *it;
};

class Beam{
 double *z_e, *z_p;
 double *s_x_e, *s_y_e, *s_x_p, *s_y_p;
 double *xb_e, *yb_e, *xb_p, *yb_p;
 int *iSize_e, *iSize_p;
 public:
 Map *fmap; //Map from file
 Map *map_e, *map_p, *eqns_e, *eqns_p;
 double *x_e, *x_p;

 void initialize(BeamParams *);
 void allocateMap(Map *, int);
 void genLinMap(int , double , double , double , double , double , double , double , double, int);

 void genICs(BeamParams *, PARTICLE_TYPE type);
 void applyM(double *, int *, double *, int *, int, int, int);
 void applyM(double *, int *, double *, int *, int, int, int, int);

 double slice(int *, double *, double *&, int *&, int, int, int, int *&);
 void mergeSlices(double *&, double *, int *, int, int, int, int *);

 void applyKick(BeamParams *, double *, double *, int , int , double , int, int);

 void collide(double *, double *, int *, int *, BeamParams *, int);


 void gf2Eqns(double *f, double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder);
 void newtonIter(double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder);

 int checkResidual(double *f, int *flag, double epsabs, int Npart);

 void Jacobian(int *flag, double *jac, double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder);
 void LUDcmp_lubksb(int *flag, double *da, double *b, double *M, int *ind, double *xi, double *xf, int *Nrow, int maxLen, int Npart, int Norder);
};

#endif /* BEAM_H */
