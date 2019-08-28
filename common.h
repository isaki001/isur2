#ifndef COMMON_H
#define COMMON_H

#define PI 3.14159265358979323846264338
#define Re 2.817940289458e-15    //! [meters]
#define Rp 1.534698e-18          //! [meters]
 
#define NCOL 6
#define PRECISION 2.2E-16

#define GCC_VERSION (__GNUC__ * 10000		      \
		     + __GNUC_MINOR__ * 100	      \
		     + __GNUC_PATCHLEVEL__)

#define NWARPS 4
#define BLOCKDIMX_COL 128
#define NSLICES 35

#include <string>

struct BeamParams{
  int iTrackOnly;        //! flag for tracking only: 1: track <>1: 2-beam
  std::string gfEqns_file_e;//! gf equations file for e-beam
  std::string gfEqns_file_p;//! gf equations file for p-beam
  std::string Me_file;   //! matrix data file for the e-beam
  int Mef;               //! flag for matrix data file for the e-beam 
  int Norder_e;          //! order of the map for the e-beam
  int Npart_e;           //! number of simulation particles in the e-beam
  int Npart_inbound_e;   //! number of simulation particles in bound in the e-beam
  std::string ICe_file;  //! file containing ICs for the e-beam
  int ICef;              //! flag for file containing ICs for the e-beam 
  std::string Mp_file;   //! matrix data file for the e-beam
  int Mpf;               //! flag for matrix data file for the e-beam
  int Norder_p;          //! order of the map for the p-beam
  int Npart_p;           //! number of simulation particles in the p-beam
  int Npart_inbound_p;   //! number of simulation particles in bound in the p-beam
  std::string ICp_file;  //! file containing ICs for the p-beam
  int ICpf;              //! flag for file containing ICs for the p-beam 
  int Niter;            //! number of iterations
  int ebunches;         //! number of ebunches
  int pbunches;         //! number of pbunches
  int NdumpOffset;       //!offset - start dumping after "offset" turns
  int Nfreq;            //! frequency of dumping DF
  int NfreqLum;          //! frequency of dumping luminosity
  int iRegime;           //! type of beam-beam effect
  double sig_x0_e;       //! rms horizontal size of the e-beam
  double sig_y0_e;       //! rms vertical size of the e-beam
  double sig_z0_e;       //! rms size of the e-beam
  double sig_dE0e;       //! rms energy spread for the e-beam
  double off_x_e;        //! initial offset of the e-beam in x
  double off_y_e;        //! initial offset of the e-beam in x
  double beta_x0_e;      //! horizontal beta* of the e-beam at IP
  double beta_y0_e;      //! vertical beta* of the e-beam at IP
  double sig_x0_p;       //! rms horizontal size of the p-beam
  double sig_y0_p;       //! rms vertical size of the p-beam
  double sig_z0_p;       //! rms size of the p-beam
  double sig_dE0p;       //! rms energy spread for the p-beam
  double off_x_p;        //! initial offset of the p-beam in x
  double off_y_p;        //! initial offset of the p-beam in x
  double beta_x0_p;      //! horizontal beta* of the p-beam at IP
  double beta_y0_p;      //! vertical beta* of the p-beam at IP
  double N_e;               //! number of particles in the e-beam
  double N_p;               //! number of particles in the p-beam
  double E_e;            //! energy of the e-beam
  double E_p;            //! energy of the p-beam
  double f_0;            //! revolution frequency
  double nu_ex;          //! betatron tune in x for the e-beam
  double nu_ey;          //! betatron tune in y for the e-beam
  double nu_ez;          //! synchrotron tune for the e-beam
  double nu_px;          //! betatron tune in x for the p-beam
  double nu_py;          //! betatron tune in y for the p-beam
  double nu_pz;          //! synchrotron tune for the p-beam
  int N;                 //! number of slices in each beam
  int isGPU;            // flag for CPU/GPU execution: 1-GPU 0-CPU
  int isSymTr;          //0 - Tracking Only, 1 - Symplectic Tracking
  int icGen;            //! IC generation method
  int coord1;           //! First uniform lattice coordinate
  int coord2;           //! Second uniform lattice coordinate
  double x_l;            //! Uniform lattice lower bound for first coordinate
  double x_u;            //! Uniform lattice upper bound for first coordinate
  double y_l;            //! Uniform lattice lower bound for second coordinate
  double y_u;            //! Uniform lattice upper bound for second coordinate
  int coordNum1;          //! Number of values for first uniform coordinate
  int coordNum2;          //! Number of values for second uniform coordinate
  double gamma_e, gamma_p, Lc, Lsl;
  double E_e0, E_p0;
  int jmaxord_e, jmaxord_p;

  double x_bound,y_bound;
  int NSympFreq;
  bool strict_freq;
  bool log_in_background;
};

enum PARTICLE_TYPE {ELECTRON, PROTON};

#endif /* COMMON_H */
