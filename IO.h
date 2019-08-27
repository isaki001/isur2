#ifndef IO_H
#define IO_H

#include<iostream>
#include<cmath>
#include<map>
#include<cstdlib>
#include<fstream>
#include<sstream>

#if GCC_VERSION > 40800
#include<thread>
#endif

#define LOG_FILE 

class IO{
 public:
  int pid;
  std::map<std::string, std::string> inputConfig;
  std::map<std::string, std::string> outputConfig;
  std::map<std::string, std::string> paths;

  IO();
  static void mergeDumps(BeamParams * , int);

  static void
    FileCreatePID(int rank = 0){
    std::ostringstream oss;
    oss << rank;
    std::ofstream out;
    std::string file = "output/" + oss.str() + ".out";

    //clean the dump files on every execution...
    system("exec rm -r ./output/dump.*");    
  }
  static void 
    IODebug(const char * error,
	  const char *filename,
	  int line,
	  bool silent=false){
    std::cerr << "Error [" << filename << ", " << line << "]: " << error << "\n";
  }  
  static void 
    IOAbort(const char * error,
	  const char *filename,
	  int line,
	  bool silent=false){
    std::cerr << "Error [" << filename << ", " << line << "]: " << error << "\nAborted!!!";
    exit(1);
  }  
  
  static void
    IOPrint(const char * msg, bool newline=true, int rank = 0){
#ifdef LOG_FILE
    std::ostringstream oss;
    oss << rank;
    std::ofstream out;
    std::string file = "output/" + oss.str() + ".out";

    out.open(file.c_str(), std::ofstream::out | std::ofstream::app);
    out.precision(16);
    out << msg;
    if(newline)
      out << std::endl;
#endif
    std::cout.precision(16);
    std::cout << msg;
    if(newline)
      std::cout << std::endl;
  }
  static void
    IOPrint(std::string msg, bool newline=true, int rank = 0){
#ifdef LOG_FILE
    std::ostringstream oss;
    oss << rank;
    std::ofstream out;
    std::string file = "output/" + oss.str() + ".out";
    out.open(file.c_str(), std::ofstream::out | std::ofstream::app);
    out.precision(16);
    out << msg;
    if(newline)
      out << std::endl;
#endif
    std::cout.precision(16);
    std::cout << msg;
    if(newline)
      std::cout << std::endl;
   }
};
  

class InputProcessing:public IO{
 public:
#if GCC_VERSION > 40800
  std::thread log_thread;
#endif
  bool pending_log, log_in_progress;

  InputProcessing(int pid = 0);
  void ReadInput(BeamParams *);
  void ReadMap(std::string, int *, double *, int *, int , int &);
  void printMap(Map *map, std::string );

  void readICs(double *&, std::string , int &, int);
  void readIC(double *&, std::string ,int,  int); //relies on the fact that Npart is determined on input file
  void dumpParticles(int , double *, int , int , int , int , std::string , std::_Ios_Openmode mode = std::ios::app, int pid = 0);
  void dumpLum(int iTurn, double Lum_total, double Lc, double Lsl, int N,
			      double xbar_e, double ybar_e, double xbar_p, double ybar_p, double sig_x_e, double sig_y_e,
			      double sig_x_p, double sig_y_p, double *mom_x_e, double *mom_y_e, double *mom_x_p, double *mom_y_p,
			      double pxbar_e, double pybar_e, double pzbar_e, double sig_px_e, double sig_py_e, double sig_pz_e,
			      double pxbar_p, double pybar_p, double pzbar_p, double sig_px_p, double sig_py_p, double sig_pz_p, std::string sFile, std::_Ios_Openmode mode);

  void dumpLum_mpi(int iTurn, double lum, std::string sFile, std::_Ios_Openmode mode = std::ios::app);

  //void dumpBeamByThread(BeamParams *bParams, double *x, int Npart, int Ncol, int iTurn, std::string ic, std::_Ios_Openmode mode = std::ios::app, int pid = 0);
  void threadFinalize();

};


  /**
   * \brief Debug macro
   */
#define __Debug(e) IO::IODebug((e), __FILE__, __LINE__)

#define __Abort(e) IO::IOAbort((e), __FILE__, __LINE__)

#define PrintlnPID(m, pid) IO::IOPrint(m, true, pid)
#define Println(m) PrintlnPID(m, this->pid)


#define Print(m) IO::IOPrint(m, false)

#endif /* IO_H */
