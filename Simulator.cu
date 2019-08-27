#include <mpi.h>
#include <iostream>
#include <iomanip>
#include "Simulator.h"

/*
#include <cuda_runtime.h>
#include "util/cudaArchUtil.h"
//#include "util/cudaDebugUtil.h"
#include "util/cudaUtil.h"
#include "util/cudaTimerUtil.h"
//#include "util/cudaMemoryUtil.h"
*/
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>


#include "GPUBeam/Map.cu"
/*
extern "C" {
#include "GPUBeam/Collide.h"
}
*/
#include "GPUBeam/Collide.h"

#define TAG  0

#define BUFSIZE 256
enum MPI_XCHANGE_TYPE{TRACKING_ONLY, EXIT, PROBE};
static std::map<MPI_XCHANGE_TYPE, std::string> MPI_DATA;

using namespace std;

Simulator::Simulator(int pid){
  this->rank = 0;
  this->deviceId = 0;
  this->num_tasks = 1;
  io = new InputProcessing;
  bParams = new BeamParams;
  beam = new Beam;
  MPI_DATA[EXIT]="EXIT";
  MPI_DATA[PROBE]="PROBE";
  MPI_DATA[TRACKING_ONLY]="TRACKING_ONLY";  
}

bool Simulator::Initialize(){
  beam->initialize(bParams);
  printf("Beam Initialized\n");
  return true;
}

//!--------------------------------------------------------------
//! Generate the map from the simulation parameters if Mef == 1.
//!--------------------------------------------------------------
void Simulator::Simulate(int argc, char **argv){
  //quad::timer::event_pair timer_node;


  if(!Initialize()){
    __Abort("Initialize Error");
  }

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  //! create the linear map for e-beam if Mef == 1
  if(bParams->Mef == 1)
  {
    bParams->Norder_e = 1;           //! linear map must be of order 1
    beam->genLinMap(1, bParams->nu_ex, bParams->nu_ey, bParams->nu_ez, bParams->beta_x0_e, bParams->beta_y0_e, bParams->gamma_e, bParams->sig_z0_e, bParams->sig_dE0e, world_rank);
  }
  

  //quad::timer::start_timer(&timer_node);
  
  //GF equations from file
  
  
  io->ReadMap(bParams->gfEqns_file_e, beam->fmap->Nrow, beam->fmap->M, beam->fmap->it, bParams->Norder_e, bParams->jmaxord_e);
  int maxLen_eqns_e = Util::maxval<int>(beam->fmap->Nrow, 6);
  
  beam->allocateMap(beam->eqns_e, maxLen_eqns_e);
  memcpy(beam->eqns_e->Nrow, beam->fmap->Nrow, 6 * sizeof(int));
  
  for(int j = 1; j <= maxLen_eqns_e; ++j){
    for(int i = 1; i <= 6; ++i){
      beam->eqns_e->M[(j - 1) * 6 + i - 1] = beam->fmap->M[(j - 1) * 6 + i - 1];
      for(int k = 1; k <= 6; ++k){
	beam->eqns_e->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k -1] = beam->fmap->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k - 1];
      }
    }
  }
  io->ReadMap(bParams->gfEqns_file_p, beam->fmap->Nrow, beam->fmap->M, beam->fmap->it, bParams->Norder_e, bParams->jmaxord_e);
  int maxLen_eqns_p = Util::maxval<int>(beam->fmap->Nrow, 6);

  beam->allocateMap(beam->eqns_p, maxLen_eqns_p);
  memcpy(beam->eqns_p->Nrow, beam->fmap->Nrow, 6 * sizeof(int));
  
  for(int j = 1; j <= maxLen_eqns_p; ++j){
    for(int i = 1; i <= 6; ++i){
      beam->eqns_p->M[(j - 1) * 6 + i - 1] = beam->fmap->M[(j - 1) * 6 + i - 1];
      for(int k = 1; k <= 6; ++k){
        beam->eqns_p->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k -1] = beam->fmap->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k - 1];
      }
    }
  }
  
  io->ReadMap(bParams->Me_file, beam->fmap->Nrow, beam->fmap->M, beam->fmap->it, bParams->Norder_e, bParams->jmaxord_e);  
  double *x_p = beam->x_p;
  int maxLen_p = 0;
  int maxLen_e = Util::maxval<int>(beam->fmap->Nrow, 6);
  beam->allocateMap(beam->map_e, maxLen_e);
  memcpy(beam->map_e->Nrow, beam->fmap->Nrow, 6 * sizeof(int));
  for(int j = 1; j <= maxLen_e; ++j){
    for(int i = 1; i <= 6; ++i){
      beam->map_e->M[(j - 1) * 6 + i - 1] = beam->fmap->M[(j - 1) * 6 + i - 1];
      for(int k = 1; k <= 6; ++k){
	beam->map_e->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k -1] = beam->fmap->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k - 1];
      }
    }
  }
  
  //!---------------------------------------------
  //! Print the map for the e-beam to the screen.
  //!---------------------------------------------
  io->printMap(beam->map_e, "out");
  //quad::timer::stop_timer(&timer_node, "ReadMap+MapInit");

  //!----------------------------------------------------------
  //! Read in the ICs for the e-beam from ICe_file (ICef <> 1)
  //! or generate it from the parameters read in (ICef == 1).
  //!----------------------------------------------------------
  double *x_e = beam->x_e;
  
  //quad::timer::start_timer(&timer_node);
  //cout<<"Read Flag:"<<bParams->ICef<<endl;
  //cout<<"Before Read/Gen Npart_e:"<<bParams->Npart_e<<endl;
  if(bParams->ICef != 1)
  {
    io->readIC(x_e, bParams->ICe_file, bParams->ebunches, NCOL);
    
  }
  else
  {
    beam->genICs(bParams, ELECTRON);
  }
  //cout<<"After Read/Gen Npart_e:"<<bParams->Npart_e<<endl;
  printf("About to Dump Particles\n");
  if(world_rank == 0)
    io->dumpParticles(bParams->ebunches, x_e, bParams->Npart_e, NCOL, bParams->Nfreq, 0, "IC_e");

  //quad::timer::stop_timer(&timer_node, "ICs");


  if(bParams->iTrackOnly != 1){
    //! create the linear map for p-beam if Mpf == 1
    if(bParams->Mpf == 1){
      bParams->Norder_p = 1;           //! linear map must be of order 1
      beam->genLinMap(0, bParams->nu_px, bParams->nu_py, bParams->nu_pz, bParams->beta_x0_p, bParams->beta_y0_p, bParams->gamma_p, bParams->sig_z0_p, bParams->sig_dE0p, world_rank);
    } 

    //!-------------------------------------------------------------------
    //! Read in the map from matrix_file and truncate it to the minimum 
    //! of Norder and the jmaxord (maximum order of the map) if Mpf <> 1.
    //!-------------------------------------------------------------------
    
    io->ReadMap(bParams->Mp_file, beam->fmap->Nrow, beam->fmap->M, beam->fmap->it, bParams->Norder_p, bParams->jmaxord_p);
    maxLen_p = Util::maxval<int>(beam->fmap->Nrow, 6);
    beam->allocateMap(beam->map_p, maxLen_p);
    memcpy(beam->map_p->Nrow, beam->fmap->Nrow, 6 * sizeof(int));

    for(int j = 1; j <= maxLen_p; ++j){
      for(int i = 1; i <= 6; ++i){
	beam->map_p->M[(j - 1) * 6 + i - 1] = beam->fmap->M[(j - 1) * 6 + i - 1];
	for(int k = 1; k <= 6; ++k){
	  beam->map_p->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k -1] = beam->fmap->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k - 1];
	}
      }
    }

    //!------------------------------------------------------------
    //! Read in the ICs for the p-beam from ICp_file (ICpf <> 1)
    //! or generate it from the parameters read in (ICpf == 1).
    //!------------------------------------------------------------
    
    if(bParams->ICpf != 1){
      io->readIC(x_p, bParams->ICp_file, bParams->pbunches, NCOL);
    }else{
      beam->genICs(bParams, PROTON);
    }
    if(world_rank == 0)
      io->dumpParticles(bParams->pbunches, x_p, bParams->Npart_p, NCOL, bParams->Nfreq, 0, "IC_p");
  }

  std::stringstream ss;
  if(bParams->iTrackOnly != 1){
    ss << "================================================================\n";
    ss << "                   BEAM-BEAM SIMULATION \n";
    ss << "================================================================\n";    
  }else{
    ss << "================================================================\n";
    ss << "                   TRACKING SIMULATION \n";
    ss << "================================================================\n";    
  }

  ss << "------------------------------------------------------\n";
  ss << "                  Simulation parameters\n";
  ss << "------------------------------------------------------\n";
  ss << "Number of map iterations       : " << bParams->Niter << "\n";
  ss << "Freqency of printing out       : " << bParams->Nfreq << "\n";

  if(bParams->iTrackOnly != 1){
    ss << "Number of slices per beam      : " << bParams->N << "\n";
    ss << "Beam-beam effect model         : " << bParams->iRegime << "\n";
    ss << "------------------------------------------------------\n";
    if (bParams->iRegime == 1) {
      ss << "                       WEAK-STRONG\n";
    }else if (bParams->iRegime == 2) {
      ss << "                       STRONG-STRONG\n";
    }else if (bParams->iRegime == 3) {
      ss << "                       NO COLLISION: compute rms\n";
    }else{
      ss << "                       NO COLLISION\n";
    }
    ss << "------------------------------------------------------\n";
  }

  ss << "\n------------------------------------------------------\n";
  ss << "                     Beam parameters\n";
  ss << "------------------------------------------------------\n";

  ss << "---------\n" ;
  ss << " e-beam \n";
  ss << "--------- \n" ;
  ss << "   Energy                      : " << bParams->E_e << "eV\n";
  ss << "   Matrix file                 : " << bParams->Me_file << "\n";
  ss << "   IC file                     : " << bParams->ICe_file << "\n";
  ss << "   Number of test particles    : " << bParams->Npart_e << "\n";
  ss << "   Map up to order             : " << bParams->jmaxord_e << "\n";
  ss << "   Requested map order         : " << bParams->Norder_e << "\n";
  ss << "   Horizontal rms size         : " << bParams->sig_x0_e << "m\n";
  ss << "   Vertical rms size           : " << bParams->sig_y0_e << "m\n";
  ss << "   Number of particles in beam : " << bParams->N_e << "\n";
  ss << "================================================================\n";

  if(bParams->iTrackOnly != 1){
    ss << "--------- \n";
    ss << " p-beam \n";
    ss << "--------- \n";
    ss << "   Energy                      : " << bParams->E_p << "eV\n";
    ss << "   Matrix file                 : " << bParams->Mp_file << "\n";
    ss << "   IC file                     : " << bParams->ICp_file << "\n";
    ss << "   Number of test particles    : " << bParams->Npart_p << "\n";
    ss << "   Map up to order             : " << bParams->jmaxord_p << "\n";
    ss << "   Requested map order         : " << bParams->Norder_p << "\n";
    ss << "   Horizontal rms size         : " << bParams->sig_x0_p << "m\n";
    ss << "   Vertical rms size           : " << bParams->sig_y0_p << "m\n";
    ss << "   Number of particles in beam : " << bParams->N_p << "\n";
  }
  ss << "================================================================\n";
  //std::cout << ss.str() << "\n"; 


  //!************************
  //! TRACKING ONLY (1 beam) 
  //!************************
  if(bParams->iTrackOnly == 1){
    char buff[BUFSIZE];
    for(int node_id = 0; node_id < num_tasks; ++node_id){
      if(node_id != 0){
	strncpy(buff, MPI_DATA[TRACKING_ONLY].c_str(), BUFSIZE);
	MPI_Send(buff, BUFSIZE, MPI_CHAR, node_id, TAG, MPI_COMM_WORLD);
      }
    }      

    if(bParams->isGPU){
      std::cout << "Executing on GPU!\n";
      double *d_x;
      int *d_Nrow = 0, *d_itM = 0;
      int *d_eqnsNrow_e = 0, *d_eqnsitM_e = 0;

      int *h_eqnsitM_e = generateMapData(beam->eqns_e, maxLen_eqns_e, bParams->Npart_e, NCOL);
      int *h_itM = generateMapData(beam->map_e, maxLen_e, bParams->Npart_e, NCOL);

      sendMetadata(maxLen_eqns_e, maxLen_e, bParams->Npart_e, NCOL, bParams);
      //Send Eqns and Map data
      for(int node_id = 0; node_id < num_tasks; ++node_id){
	if(node_id != 0){
	  MPI_Send(h_eqnsitM_e, maxLen_eqns_e * NCOL * (NCOL + 2), MPI_INT, node_id, TAG, MPI_COMM_WORLD);
	  MPI_Send(h_itM, maxLen_e * NCOL * (NCOL + 2), MPI_INT, node_id, TAG, MPI_COMM_WORLD);
	  MPI_Send(beam->eqns_e->Nrow, NCOL, MPI_INT, node_id, TAG, MPI_COMM_WORLD);
	  MPI_Send(beam->map_e->Nrow, NCOL, MPI_INT, node_id, TAG, MPI_COMM_WORLD);
	}
      }

      //Partition particles between nodes
      double *h_x = 0;
      int Npart = 0;
      int numParticlesPerNode = bParams->Npart_e / num_tasks;
      for(int node_id = 0; node_id < num_tasks; ++node_id){
	int offset = node_id * numParticlesPerNode;
	int count = numParticlesPerNode;
	if(node_id == num_tasks - 1){
	  count = bParams->Npart_e - offset;
	}

	double *hx = new double[count * (NCOL)];

	for(int ii = 0; ii < NCOL; ++ii){
	  memcpy(hx + ii * count, x_e + ii * bParams->Npart_e + offset, sizeof(double) * count);
	}
	if(node_id != 0){
	  MPI_Send(hx, count * NCOL, MPI_DOUBLE, node_id, TAG, MPI_COMM_WORLD);
	}else{
	  h_x = hx;
	  Npart = count;
	}
      }

      initDeviceMemory(d_eqnsitM_e, h_eqnsitM_e,
		       d_eqnsNrow_e, beam->eqns_e->Nrow,
		       d_x, h_x,
		       maxLen_eqns_e, Npart, NCOL);

      initDeviceMemory(d_itM, h_itM,
		       d_Nrow, beam->map_e->Nrow,
		       d_x, h_x,
		       maxLen_e, Npart, NCOL);

      double *dOpx = 0;
      QuadDebug(cudaMalloc((void **)&dOpx, sizeof(double) * Npart * NCOL));
      int *dOutOfBound = 0;
      QuadDebug(cudaMalloc((void **)&dOutOfBound, sizeof(int) * Npart));
      thrust::device_ptr<int> dev_ptr(dOutOfBound);    
      thrust::fill(dev_ptr, dev_ptr + Npart, (int) 0);

      //quad::timer::event_pair timer0;
      //quad::timer::start_timer(&timer0);
    
      double time = 0;
      for(int iTurn = 1; iTurn <= bParams->Niter; ++iTurn){	
	double exec_time = applyMapGPU(dOutOfBound,
				       d_itM, d_eqnsitM_e, 
				       d_x, dOpx,
				       beam->map_e->Nrow, beam->eqns_e->Nrow, 
				       h_x, 
				       maxLen_e, maxLen_eqns_e, 
				       Npart, NCOL, bParams, iTurn);
	time += exec_time;
	//dumpBeamByThread(this, d_x, h_x, Npart, NCOL, iTurn, "dump.ebeam", std::ios::app, 1);
      }
      io->threadFinalize();
      //quad::timer::stop_timer(&timer0, "GPU Tracking");

      //ss.str("");
      //ss << "Tracking took " << time/bParams->Niter << " ms  per turn in " << hostname << " (Rank = " << rank << ")\n" ;
      //std::cout << ss.str();

      cudaFree(dOpx);

      //io->dumpParticles(h_x, Npart, NCOL, bParams->Nfreq, bParams->Niter, "dump.ebeam");
    }else{
      std::cout << "Executing on CPU!\n";
      //quad::timer::event_pair timer0;
      //quad::timer::start_timer(&timer0);
    
      int iTurn = 1;
      double *xi = new double[NCOL * bParams->Npart_e];
      for(iTurn = 1; iTurn <= bParams->Niter; ++iTurn){
	memcpy(xi, x_e, NCOL * bParams->Npart_e * sizeof(double));
	beam->applyM(beam->map_e->M, beam->map_e->it, x_e, beam->map_e->Nrow, maxLen_e, bParams->Npart_e, bParams->Norder_e);
      
	//TODO - xi to be updated every iteration
	if(bParams->isSymTr && bParams->NSympFreq > 0 && (iTurn % bParams->NSympFreq == 0)){
	  beam->newtonIter(beam->eqns_e->M, beam->eqns_e->it, xi, x_e, beam->eqns_e->Nrow, maxLen_eqns_e, bParams->Npart_e, bParams->Norder_e);
	}

	if(bParams->Nfreq > 0 && (iTurn % bParams->Nfreq) == 0){
	  std::stringstream ss;
	  ss << std::setprecision(16);
	  ss << iTurn << " turns finished\n";
	  std::cout << ss.str() << "\n";
	  //io->dumpParticles(bParams->ebunches, x_e, bParams->Npart_e, NCOL, bParams->Nfreq, iTurn, "dump.ebeam");	
	}
      }
      //quad::timer::stop_timer(&timer0, "CPU applyMap");
      //io->dumpParticles(bParams->ebunches, x_e, bParams->Npart_e, NCOL, bParams->Nfreq, bParams->Niter, "dump.ebeam");	
    }
  }else{
    //!*************************
    //! TRACKING AND COLLISION
    //!*************************    

    //quad::timer::start_timer(&timer_node);
    printf("TrackingWithCollision\n");
    TrackingWithCollision(maxLen_e, maxLen_p);
    //quad::timer::stop_timer(&timer_node, "Tracking");
  }
}



void Simulator::TrackingWithCollision(int &maxLen_e, int &maxLen_p){
  
  quad::timer::event_pair timer_pretrack;
  quad::timer::start_timer(&timer_pretrack);

  double *x_e = beam->x_e;
  double *x_p = beam->x_p;

  int Npart_e = bParams->Npart_e;

  for(int b = 0; b < bParams->ebunches; b++) 
    for(int i = Npart_e * NCOL; i < Npart_e * (NCOL + 2); i++)
      x_e[b * Npart_e * (NCOL + 2) + i] = 0;
		
  double *dx_e;
  int maxLen_eqns_e = Util::maxval<int>(beam->fmap->Nrow, 6);
  int *h_itM_e = generateMapData(beam->map_e, maxLen_e, bParams->Npart_e, NCOL);
  int *h_eqnsitM_e = generateMapData(beam->eqns_e, maxLen_eqns_e, bParams->Npart_e, NCOL);
  int *d_Nrow_e = 0, *d_itM_e = 0;
  int *d_eqnsNrow_e = 0, *d_eqnsitM_e = 0;

  initDeviceMemory_bunch(d_eqnsitM_e, h_eqnsitM_e,
                         d_eqnsNrow_e, beam->eqns_e->Nrow,
                         dx_e, x_e,
                         maxLen_eqns_e, Npart_e, NCOL, bParams->pbunches, 0);
  
  initDeviceMemory_bunch(d_itM_e, h_itM_e,
                         d_Nrow_e, beam->map_e->Nrow,
                         dx_e, x_e,
                         maxLen_e, Npart_e, NCOL, bParams->pbunches, 0);

  /////////////////////////////////////////////////////////////////////////////

  int Npart_p = bParams->Npart_p;

  for(int b = 0; b < bParams->pbunches; b++) 
    for(int i = Npart_p * NCOL; i < Npart_p * (NCOL + 2); i++)
      x_p[b * Npart_p * (NCOL + 2) + i] = 0;
        
  

  double *dx_p;

  int maxLen_eqns_p = Util::maxval<int>(beam->fmap->Nrow, 6);
  int *h_itM_p = generateMapData(beam->map_p, maxLen_p, bParams->Npart_p, NCOL);
  int *h_eqnsitM_p = generateMapData(beam->eqns_p, maxLen_eqns_p, bParams->Npart_p, NCOL);
  int *d_Nrow_p = 0, *d_itM_p = 0;
  int *d_eqnsNrow_p = 0, *d_eqnsitM_p = 0;

  initDeviceMemory_bunch(d_eqnsitM_p, h_eqnsitM_p,
                         d_eqnsNrow_p, beam->eqns_p->Nrow,
                         dx_p, x_p,
                         maxLen_eqns_p, Npart_p, NCOL, bParams->pbunches, 1);

  initDeviceMemory_bunch(d_itM_p, h_itM_p,
                         d_Nrow_p, beam->map_p->Nrow,
                         dx_p, x_p,
                         maxLen_p, Npart_p, NCOL, bParams->pbunches, 1);

  int e_schedule[(bParams->pbunches) * (bParams->ebunches)], p_schedule[(bParams->pbunches) * (bParams->ebunches)], e_i = 0, p_i = 1;

  for(int i = 0; i < (bParams->pbunches) * (bParams->ebunches); i++){
    if (e_i < (bParams->ebunches))
      e_schedule[i] = e_i;
    else  {
      e_i = 0;
      e_schedule[i] = e_i;	
    }
    e_i++;

    if (p_i < bParams->pbunches)
      p_schedule[i] = p_i;
    else  {
      p_i = 0;
      p_schedule[i] = p_i;
    }
    p_i++;
		
    //std::cout<<"e_schedule[i] is : "<<e_schedule[i]<<" -- "<<"p_schedule[i] is : "<<p_schedule[i]<<"\n";
  }
	
  quad::timer::stop_timer(&timer_pretrack, "Pre_Tracking");

	
  //MPI_Init(NULL, NULL);
	
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Barrier(MPI_COMM_WORLD);
  quad::timer::event_pair gpu_alloc;
  quad::timer::start_timer(&gpu_alloc);
	
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  int num_bunch_p = bParams->pbunches, num_gpu = world_size;
  int num_bunch_e = bParams->ebunches;
  int extra_bunches_e = bParams->ebunches - (bParams->ebunches/world_size) * world_size; 
  int extra_bunches_p = bParams->pbunches - (bParams->pbunches/world_size) * world_size; 
  int bunch_no = 0, max_bunches_per_gpu_e = 0, max_bunches_per_gpu_p = 0;
  int gpu_of_bunch_e[bParams->ebunches], gpu_of_bunch_p[bParams->pbunches];
  int bunch_count_of_gpu_e[world_size]; 
  int bunch_count_of_gpu_p[world_size]; 
  int starting_bunch_of_gpu_e[num_gpu];
  int starting_bunch_of_gpu_p[num_gpu];

  
  if(extra_bunches_e > 0)
    max_bunches_per_gpu_e = (num_bunch_e/num_gpu) + 1;
  else
    max_bunches_per_gpu_e = num_bunch_e/num_gpu;

  for(int gpu_no = 0; gpu_no < extra_bunches_e; gpu_no++){
    bunch_count_of_gpu_e[gpu_no] = 0;
    for(int j = 0; j < (num_bunch_e/num_gpu) + 1; j++){
      if(j == 0) 
        starting_bunch_of_gpu_e[gpu_no] = bunch_no;
      
      gpu_of_bunch_e[bunch_no] = gpu_no;
      bunch_count_of_gpu_e[gpu_no]++;
      bunch_no++;
    }
  }

  for(int gpu_no = extra_bunches_e; gpu_no < num_gpu; gpu_no++){
    bunch_count_of_gpu_e[gpu_no] = 0;
    for(int j = 0; j < (num_bunch_e/num_gpu); j++){
      if(j == 0) 
        starting_bunch_of_gpu_e[gpu_no] = bunch_no;

      gpu_of_bunch_e[bunch_no] = gpu_no;
      bunch_count_of_gpu_e[gpu_no]++;
      bunch_no++;
    }
  } 

  bunch_no = 0;
  if(extra_bunches_p > 0)
    max_bunches_per_gpu_p = (num_bunch_p/num_gpu) + 1;
  else
    max_bunches_per_gpu_p = num_bunch_p/num_gpu;

  for(int gpu_no = 0; gpu_no < extra_bunches_p; gpu_no++){
      bunch_count_of_gpu_p[gpu_no] = 0;
      for(int j = 0; j < (num_bunch_p/num_gpu) + 1; j++)
        {
          if(j == 0) 
            starting_bunch_of_gpu_p[gpu_no] = bunch_no;            
          gpu_of_bunch_p[bunch_no] = gpu_no;
          bunch_count_of_gpu_p[gpu_no]++;
          bunch_no++;
        }
	
    }

  for(int gpu_no = extra_bunches_p; gpu_no < num_gpu; gpu_no++)
    {
      bunch_count_of_gpu_p[gpu_no] = 0;
      for(int j = 0; j < (num_bunch_p/num_gpu); j++){
        if(j == 0) 
          starting_bunch_of_gpu_p[gpu_no] = bunch_no;
                        
        gpu_of_bunch_p[bunch_no] = gpu_no;
        bunch_count_of_gpu_p[gpu_no]++;
        bunch_no++;
      }
    } 
  
  if(world_rank == 0)
    {
    cout<<"E ALLOCATION"<<endl;
    cout<<"Bunch | GPU"<<endl;
    cout<<"==========="<<endl;
    for(int i=0; i<num_bunch_e; i++)
    cout<<i<<":"<<gpu_of_bunch_e[i]<<endl;
    cout<<"==========="<<endl;
    }
    if(world_rank == 0)
    {
    cout<<"P ALLOCATION"<<endl;
    cout<<"Bunch | GPU"<<endl;
    cout<<"==========="<<endl;
    for(int i=0; i<num_bunch_p; i++)
    cout<<i<<":"<<gpu_of_bunch_p[i]<<endl;
    cout<<"==========="<<endl;
    }
	
	
	
  double *x_e_mp, *x_p_mp;
  double *dx_e_mp, *dx_p_mp, *dOpx_e_mp, *dOpx_p_mp;

  int *dOutOfBound_e_mp, *dOutOfBound_p_mp;
  /*
  //for(int i = 0; i < num_gpu; i++) {
  //	if(i == world_rank){
  //std::cout<<"Starting bunch of gpu num "<<i<<" is "<<starting_bunch_of_gpu[i]<<"\n";
  //std::cout<<"Bunch count of GPU is "<<bunch_count_of_gpu[i]<<"\n";
  */
			
			 
  x_e_mp = new double[bunch_count_of_gpu_e[world_rank] * Npart_e * (NCOL + 2)];
  x_p_mp = new double[bunch_count_of_gpu_p[world_rank] * Npart_p * (NCOL + 2)];
	
  //cout<<"starting_bunch_of_gpu_e["<<world_rank<<"] * Npart_e * (NCOL + 2):"<<starting_bunch_of_gpu_e[world_rank] * Npart_e * (NCOL + 2)<<endl;
  memcpy(x_e_mp, x_e + starting_bunch_of_gpu_e[world_rank] * Npart_e * (NCOL + 2), sizeof(double) * bunch_count_of_gpu_e[world_rank] * Npart_e * (NCOL + 2));
  memcpy(x_p_mp, x_p + starting_bunch_of_gpu_p[world_rank] * Npart_p * (NCOL + 2), sizeof(double) * bunch_count_of_gpu_p[world_rank] * Npart_p * (NCOL + 2));

  cudaMalloc((void **)&dx_e_mp, sizeof(double) * (bunch_count_of_gpu_e[world_rank]) * Npart_e * (NCOL + 2));
  cudaMalloc((void **)&dOpx_e_mp, sizeof(double) * (bunch_count_of_gpu_e[world_rank]) * Npart_e * (NCOL + 2));
  cudaMalloc((void **)&dOutOfBound_e_mp, sizeof(double) * (bunch_count_of_gpu_e[world_rank]) * Npart_e);
	
  cudaMemcpy(dx_e_mp, x_e_mp, sizeof(double) * (bunch_count_of_gpu_e[world_rank]) * (NCOL + 2) * bParams->Npart_e, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&dx_p_mp, sizeof(double) * (bunch_count_of_gpu_p[world_rank]) * Npart_p * (NCOL + 2));
  cudaMalloc((void **)&dOpx_p_mp, sizeof(double) * (bunch_count_of_gpu_p[world_rank]) * Npart_p * (NCOL + 2));
  cudaMalloc((void **)&dOutOfBound_p_mp, sizeof(double) * (bunch_count_of_gpu_p[world_rank]) * Npart_p);

  cudaMemcpy(dx_p_mp, x_p_mp, sizeof(double) * (bunch_count_of_gpu_p[world_rank]) * (NCOL + 2) * bParams->Npart_p, cudaMemcpyHostToDevice);

  thrust::device_ptr<int> dev_ptr_e(dOutOfBound_e_mp);
  thrust::fill(dev_ptr_e, dev_ptr_e + (bunch_count_of_gpu_e[world_rank]) * Npart_e, (int) 0);

  thrust::device_ptr<int> dev_ptr_p(dOutOfBound_p_mp);
  thrust::fill(dev_ptr_p, dev_ptr_p + (bunch_count_of_gpu_p[world_rank]) * Npart_p, (int) 0);
			
	
  int total_ij = 0, current_e_schedule[num_bunch_e], current_p_schedule[num_bunch_e], bunch_offset = 0;
  int sub_e_schedule[num_gpu], sub_p_schedule[num_gpu], gpu_occ_e[num_gpu], gpu_occ_p[num_gpu], final_p_bunch_of_gpu[num_gpu], final_e_bunch_of_gpu[num_gpu], gpu_num_p, gpu_num_e, rem_bunch = 0, remained_e_bunch[num_gpu], remained_p_bunch[num_gpu], p_bunch_num, e_bunch_num, opp_e_bunch[num_bunch_e], opp_p_bunch[num_bunch_p];
	
	
  int my_e_bunch_offset, my_p_bunch_offset;
  double *dx_e_cur_bunch, *dOpx_e_cur_bunch, *dx_p_cur_bunch, *dOpx_p_cur_bunch, *x_e_cur_bunch, *x_p_cur_bunch;

  int *dOutOfBound_e_cur_bunch, *dOutOfBound_p_cur_bunch;

  int dout_e = 0, dout_p = 0;

  MPI_Request myRequest_recv_e[10], myRequest_send_p[10], myRequest_send_Sumary[3], myRequest_recv_Sumary[3];
  MPI_Status status_recv_e[10], status_send_p[10], status_recv_Sumary[3], status_send_Sumary[3];

  double sig_x_p_rcvd, sig_y_p_rcvd, xbar_p_rcvd, ybar_p_rcvd;
  double sig_x, sig_y, arg, Lum = 0.0, Lum_rcvd, Lum_total = 0.0;

  double xbar_e = 0, ybar_e = 0, zbar_e = 0;
  double sig_x_e = 0, sig_y_e = 0, sig_z_e = 0;
  double *mom_x_e = 0, *mom_y_e = 0, *mom_z_e = 0;
  mom_x_e = new double[2];
  mom_y_e = new double[2];
  mom_z_e = new double[2];
  double pxbar_e = 0, pybar_e = 0, pzbar_e = 0, sig_px_e = 0, sig_py_e = 0, sig_pz_e = 0;

  double xbar_p = 0, ybar_p = 0, zbar_p = 0;
  double sig_x_p = 0, sig_y_p = 0, sig_z_p = 0;
  double *mom_x_p = 0, *mom_y_p = 0, *mom_z_p = 0;
  mom_x_p = new double[2];
  mom_y_p = new double[2];
  mom_z_p = new double[2];
  double momxp0;
  double momxp1;
  double momyp0;
  double momyp1;
  double *mom_x_p_rcvd = 0, *mom_y_p_rcvd = 0;
  double pxbar_p = 0, pybar_p = 0, pzbar_p = 0, sig_px_p = 0, sig_py_p = 0, sig_pz_p = 0;
  mom_x_p_rcvd = new double[2];
  mom_y_p_rcvd = new double[2];
  double sig_px_p_rcvd = 0, sig_py_p_rcvd=0;
  int lum_turn = 0, opp_e_bunch_gpu, opp_p_bunch_gpu, opp_e_bunch_var, opp_p_bunch_var;

  //quad::timer::event_pair timer_node;
  //quad::timer::start_timer(&timer_node);
  MPI_Barrier(MPI_COMM_WORLD);
  quad::timer::stop_timer(&gpu_alloc, "gpu_alloc");
  quad::timer::event_pair timer_collide;
  quad::timer::event_pair timer_post_col;//start timer
  int GPU_COLLIDE = 0;
  int temp_post_col = 0;
  int temp_post_col2 = 0;
  int temp_col = 0;
  int POST_COLLIDE = 0;int POST_COLLIDE2 = 0;
  for(int iTurn = 1; iTurn <= bParams->Niter; ++iTurn){
    total_ij = 0;
    for(int i = 0; i < num_bunch_p; i++){
      //quad::timer::event_pair timer_iter;
      //quad::timer::start_timer(&timer_iter);

      Lum = 0.0;
      for(int j = 0; j < num_bunch_e; j++){
        current_e_schedule[j] = e_schedule[total_ij];
        current_p_schedule[j] = p_schedule[total_ij];
        opp_p_bunch[current_e_schedule[j]] = current_p_schedule[j]; //this keep track of the bunches, not caring about order of collisions
        opp_e_bunch[current_p_schedule[j]] = current_e_schedule[j];	//so if you go to 6th index of opp_e_bunch, you will see which bunch e6 will collide with in this sub-schedule
        total_ij++;
      }
        	
      for(int timestamp = 0; timestamp < max_bunches_per_gpu_p; timestamp++){
        //quad::timer::event_pair timer_schedule;
        //quad::timer::start_timer(&timer_schedule);
				
        bunch_offset = timestamp;
        //sub-schedule			
        for(int j = 0; j < num_gpu; j++){
          sub_e_schedule[j] = -1;
          sub_p_schedule[j] = -1;
          gpu_occ_e[j] = -1;
          gpu_occ_p[j] = -1;
          final_e_bunch_of_gpu[j] = -1;
          final_p_bunch_of_gpu[j] = -1;
					
          //we grab what collision we want to do for each GPU without looking at resource contention
          if(bunch_offset < num_bunch_e && timestamp < bunch_count_of_gpu_e[j]){
              
            sub_e_schedule[j] = current_e_schedule[bunch_offset];					
            if(timestamp == 0){
              final_e_bunch_of_gpu[j] = current_e_schedule[bunch_offset];
              gpu_occ_e[j] = 1;
            }

            sub_p_schedule[j] = current_p_schedule[bunch_offset];
          }
            
          bunch_offset += bunch_count_of_gpu_e[j]; // --> the bunch in this position should be pushed next to the schedule slot
        }

        //Subscheduling goes here below
        if(timestamp == 0){
          for(int j = 0; j < num_gpu; j++){
            if(sub_e_schedule[j] != -1 && sub_p_schedule[j] != -1){
              p_bunch_num = sub_p_schedule[j];	//assing P bunch
              gpu_num_p = gpu_of_bunch_p[p_bunch_num];	//get its gpu
              e_bunch_num = sub_e_schedule[j];	//assign e bunch
              gpu_num_e = gpu_of_bunch_e[e_bunch_num];	//get its gpu

              if(gpu_occ_p[gpu_num_p] != 1) //occupancy 1 means GPU busy, if not busy
                {
                  final_p_bunch_of_gpu[gpu_num_p] = sub_p_schedule[j];	//mark the final p bunch for the sub-schedule
                  gpu_occ_p[gpu_num_p] = 1;								//mark GPU for p, as busy
                }
              else													//if GPU is busy for some other p
                {
                  remained_e_bunch[rem_bunch] = sub_e_schedule[j];		//place both e and p in queue 
                  remained_p_bunch[rem_bunch] = sub_p_schedule[j];
                  gpu_occ_e[gpu_num_e] = -1;								//free up its e part
                  //gpu_occ_p[gpu_num_p] = -1;
                  final_e_bunch_of_gpu[gpu_num_e] = -1;					//free up its e part
                  //final_p_bunch_of_gpu[j] = -1;
                  rem_bunch++;											//counter for how many bunches in queue
                }
            }
          }
        }
        else{
          //we first look at the bunches in the queue remaining bunches
          for(int rb = 0; rb < rem_bunch; rb++){
            e_bunch_num = remained_e_bunch[rb];
            p_bunch_num = remained_p_bunch[rb];
            gpu_num_p = gpu_of_bunch_p[p_bunch_num];
            gpu_num_e = gpu_of_bunch_e[e_bunch_num];
            final_p_bunch_of_gpu[gpu_num_p] = p_bunch_num;
            gpu_occ_p[gpu_num_p] = 1;
            final_e_bunch_of_gpu[gpu_num_e] = e_bunch_num;
            gpu_occ_e[gpu_num_e] = 1;
          }
          rem_bunch = 0;

          for(int j = 0; j < num_gpu; j++){
            if(sub_e_schedule[j] != -1 && sub_p_schedule[j] != -1){
              p_bunch_num = sub_p_schedule[j];
              gpu_num_p = gpu_of_bunch_p[p_bunch_num];
              e_bunch_num = sub_e_schedule[j];
              gpu_num_e = gpu_of_bunch_e[e_bunch_num];
                  
              if(gpu_occ_p[gpu_num_p] != 1 && gpu_occ_e[gpu_num_e] != 1) {
                final_p_bunch_of_gpu[gpu_num_p] = sub_p_schedule[j];
                gpu_occ_p[gpu_num_p] = 1;
                final_e_bunch_of_gpu[gpu_num_e] = sub_e_schedule[j];
                gpu_occ_e[gpu_num_e] = 1;
              }
              else {
                remained_e_bunch[rem_bunch] = sub_e_schedule[j];
                remained_p_bunch[rem_bunch] = sub_p_schedule[j];       
                rem_bunch++;
              }
            }
          }
        }
			
						
        if(final_e_bunch_of_gpu[world_rank] != -1){
          my_e_bunch_offset = final_e_bunch_of_gpu[world_rank] - starting_bunch_of_gpu_e[world_rank];
          dOutOfBound_e_cur_bunch = &dOutOfBound_e_mp[my_e_bunch_offset * Npart_e];
          dx_e_cur_bunch = &dx_e_mp[my_e_bunch_offset * Npart_e * (NCOL + 2)];
          dOpx_e_cur_bunch = &dOpx_e_mp[my_e_bunch_offset * Npart_e * (NCOL + 2)]; 
          x_e_cur_bunch = &x_e_mp[my_e_bunch_offset * Npart_e * (NCOL + 2)];
          opp_p_bunch_var = opp_p_bunch[final_e_bunch_of_gpu[world_rank]];
          opp_p_bunch_gpu = gpu_of_bunch_p[opp_p_bunch[final_e_bunch_of_gpu[world_rank]]];
				
        }
        else{
          my_e_bunch_offset = 0;
        }
			
        if(final_p_bunch_of_gpu[world_rank] != -1){
          my_p_bunch_offset = final_p_bunch_of_gpu[world_rank] - starting_bunch_of_gpu_p[world_rank];
          dOutOfBound_p_cur_bunch = &dOutOfBound_p_mp[my_p_bunch_offset * Npart_p];
          dx_p_cur_bunch = &dx_p_mp[my_p_bunch_offset * Npart_p * (NCOL + 2)];
          dOpx_p_cur_bunch = &dOpx_p_mp[my_p_bunch_offset * Npart_p * (NCOL + 2)];
          x_p_cur_bunch = &x_p_mp[my_p_bunch_offset * Npart_p * (NCOL + 2)];
          opp_e_bunch_var = opp_e_bunch[final_p_bunch_of_gpu[world_rank]];
          opp_e_bunch_gpu = gpu_of_bunch_e[opp_e_bunch[final_p_bunch_of_gpu[world_rank]]];
        }
        else{
          my_p_bunch_offset = 0;
        }

        thrust::device_ptr<int> ptr;
        
        if(final_e_bunch_of_gpu[world_rank] != -1){
          applyMapGPU(dOutOfBound_e_cur_bunch,
                      d_itM_e, d_eqnsitM_e,
                      dx_e_cur_bunch, dOpx_e_cur_bunch,
                      beam->map_e->Nrow, beam->eqns_e->Nrow,
                      x_e_cur_bunch,
                      maxLen_e, maxLen_eqns_e,
                      Npart_e, NCOL, bParams, iTurn);

          ptr = thrust::device_pointer_cast(dOutOfBound_e_cur_bunch);
          dout_e = thrust::reduce(ptr, ptr + Npart_e, (int) 0, thrust::plus<int>());

          bParams->Npart_inbound_e = bParams->Npart_e - dout_e;
        }
				
        if(final_p_bunch_of_gpu[world_rank] != -1){
          
          applyMapGPU(dOutOfBound_p_cur_bunch,
                      d_itM_p, d_eqnsitM_p,
                      dx_p_cur_bunch, dOpx_p_cur_bunch,
                      beam->map_p->Nrow, beam->eqns_p->Nrow,
                      x_p_cur_bunch,
                      maxLen_p, maxLen_eqns_p,
                      Npart_p, NCOL, bParams, iTurn);

          ptr = thrust::device_pointer_cast(dOutOfBound_p_cur_bunch);
          dout_p = thrust::reduce(ptr, ptr + Npart_p, (int) 0, thrust::plus<int>());
          bParams->Npart_inbound_p = bParams->Npart_p - dout_p;
                
        }
            
        MPI_Barrier(MPI_COMM_WORLD);
        
        quad::timer::start_timer(&timer_collide);

        //second timer to measure individual collisions
        quad::timer::event_pair ind_col;
        quad::timer::event_pair timer_post_col;
        quad::timer::event_pair timer_post_col2;

        if(bParams->isGPU){
          Collide *collide = new Collide;
          quad::timer::start_timer(&ind_col);
          collide->collide(lum_turn, 
                           &dOutOfBound_e_mp[my_e_bunch_offset * Npart_e], 
                           &dOutOfBound_p_mp[my_p_bunch_offset * Npart_p], 
                           &dx_e_mp[my_e_bunch_offset * Npart_e * (NCOL + 2)],
                           &dx_p_mp[my_p_bunch_offset * Npart_p * (NCOL + 2)],
                           bParams, 
                           final_e_bunch_of_gpu[world_rank],
                           opp_p_bunch_var,
                           opp_p_bunch_gpu, 
                           final_p_bunch_of_gpu[world_rank],
                           opp_e_bunch_var,
                           opp_e_bunch_gpu);
          
          quad::timer::stop_timer(&ind_col, "ind_col");
        }
        else
          __Abort("Multi-CPU not Implemented!!!\n");		
              		
        MPI_Barrier(MPI_COMM_WORLD);
        temp_col = stop_timer_returntime(&timer_collide, "GPU collide");
        GPU_COLLIDE = temp_col + GPU_COLLIDE;
        MPI_Barrier(MPI_COMM_WORLD);
        quad::timer::start_timer(&timer_post_col);	
        int temp_post_col;
        quad::timer::start_timer(&timer_post_col2);
            
        if(final_e_bunch_of_gpu[world_rank] != -1){ 
              
          computeRMS(&dx_e_mp[my_e_bunch_offset * Npart_e * (NCOL + 2)], 
                     bParams->Npart_e,
                     xbar_e,   ybar_e,   zbar_e, 
                     sig_x_e,  sig_y_e,  sig_z_e,
                     mom_x_e,  mom_y_e,  mom_z_e,
                     pxbar_e,  pybar_e,  pzbar_e, 
                     sig_px_e, sig_py_e, sig_pz_e, 
                     bParams->Npart_inbound_e);
              
          cudaMemcpy(x_e_mp, dx_e_mp, 
                     sizeof(double)*(bunch_count_of_gpu_e[world_rank])*(NCOL+2)*bParams->Npart_e, cudaMemcpyDeviceToHost);
        }

        if(final_p_bunch_of_gpu[world_rank] != -1){
          computeRMS(&dx_p_mp[my_p_bunch_offset * Npart_p * (NCOL + 2)], 
                     bParams->Npart_p,
                     xbar_p,   ybar_p,   zbar_p, 
                     sig_x_p,  sig_y_p,  sig_z_p,
                     mom_x_p,  mom_y_p,  mom_z_p,
                     pxbar_p,  pybar_p,  pzbar_p, 
                     sig_px_p, sig_py_p, sig_pz_p, 
                     bParams->Npart_inbound_p);
          if(world_rank == 0)
            printf("Computerd Sig_x_p:%f\n", sig_x_p);
          cudaMemcpy(x_p_mp, dx_p_mp, 
                     sizeof(double)*(bunch_count_of_gpu_p[world_rank])*(NCOL+2)*bParams->Npart_p, cudaMemcpyDeviceToHost);
        }
            
        MPI_Barrier(MPI_COMM_WORLD);			
        temp_post_col2 = stop_timer_returntime(&timer_post_col2, "Post Collide2");
        POST_COLLIDE2 = temp_post_col2 + POST_COLLIDE2;

            
        if(final_p_bunch_of_gpu[world_rank] != -1){
          if(gpu_of_bunch_e[opp_e_bunch[final_p_bunch_of_gpu[world_rank]]] != world_rank){
            if(world_rank == 0)
              printf("sending sig_x_p:%f\n", sig_x_p);
            MPI_Isend(&sig_x_p,
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 0, MPI_COMM_WORLD, &myRequest_send_p[0]);
                      
            MPI_Isend(&sig_y_p, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_send_p[1]);
              
            MPI_Isend(&xbar_p, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_send_p[2]);
              
            MPI_Isend(&ybar_p, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 3,  MPI_COMM_WORLD, &myRequest_send_p[3]);
              
            MPI_Isend(&sig_px_p, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_send_p[4]);
              
            MPI_Isend(&sig_py_p, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_send_p[5]);
              
            MPI_Isend(&momxp0, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_send_p[6]);

            MPI_Isend(&momxp1, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_send_p[7]);

            MPI_Isend(&momyp0, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_send_p[8]);

            MPI_Isend(&momyp1, 
                      1, MPI_DOUBLE, opp_e_bunch_gpu, 9, MPI_COMM_WORLD, &myRequest_send_p[9]);
                    
          }
        }
				
        if(final_e_bunch_of_gpu[world_rank] != -1){
          if(gpu_of_bunch_p[opp_p_bunch[final_e_bunch_of_gpu[world_rank]]] != world_rank){
            
            MPI_Irecv(&sig_x_p_rcvd, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 0, MPI_COMM_WORLD, &myRequest_recv_e[0]);
              
            MPI_Irecv(&sig_y_p_rcvd, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 1, MPI_COMM_WORLD, &myRequest_recv_e[1]);

            MPI_Irecv(&xbar_p_rcvd, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 2, MPI_COMM_WORLD, &myRequest_recv_e[2]);

            MPI_Irecv(&ybar_p_rcvd, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 3, MPI_COMM_WORLD, &myRequest_recv_e[3]);
              
            MPI_Irecv(&sig_px_p_rcvd, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 4, MPI_COMM_WORLD, &myRequest_recv_e[4]);
              
            MPI_Irecv(&sig_py_p_rcvd, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 5, MPI_COMM_WORLD, &myRequest_recv_e[5]);

            MPI_Irecv(&momxp0, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 6, MPI_COMM_WORLD, &myRequest_recv_e[6]);
              
            MPI_Irecv(&momxp1, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 7, MPI_COMM_WORLD, &myRequest_recv_e[7]);
              
            MPI_Irecv(&momyp0, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 8, MPI_COMM_WORLD, &myRequest_recv_e[8]);
              
            MPI_Irecv(&momyp1, 1, MPI_DOUBLE, 
                      opp_p_bunch_gpu, 9, MPI_COMM_WORLD, &myRequest_recv_e[9]);
                    
            mom_x_p_rcvd[0] = momxp0;
            mom_x_p_rcvd[1] = momxp1;
            mom_y_p_rcvd[0] = momyp0;
            mom_y_p_rcvd[1] = momyp1;

          }
          else{
            if(world_rank == 0)
              printf("storing sig ps to received vars\n");
            sig_x_p_rcvd = sig_x_p;
            sig_y_p_rcvd = sig_y_p;
            xbar_p_rcvd = xbar_p;
            ybar_p_rcvd = ybar_p;
            sig_px_p_rcvd = sig_px_p;
            sig_py_p_rcvd = sig_py_p;
            mom_x_p_rcvd = mom_x_p;
            mom_y_p_rcvd = mom_y_p;
          }
        }

        if(final_p_bunch_of_gpu[world_rank] != -1){
          if(gpu_of_bunch_e[opp_e_bunch[final_p_bunch_of_gpu[world_rank]]] != world_rank)
            MPI_Waitall(10, myRequest_send_p, status_send_p);        
        }

        if(final_e_bunch_of_gpu[world_rank] != -1){
          if(gpu_of_bunch_p[opp_p_bunch[final_e_bunch_of_gpu[world_rank]]] != world_rank)
            MPI_Waitall(10, myRequest_recv_e, status_recv_e);
        }

        if(final_e_bunch_of_gpu[world_rank] != -1){
          sig_x = sqrt(sig_x_e * sig_x_e + sig_x_p_rcvd * sig_x_p_rcvd);
          sig_y = sqrt(sig_y_e * sig_y_e + sig_y_p_rcvd * sig_y_p_rcvd);
          arg = -0.50*pow(((xbar_p_rcvd-xbar_e)/sig_x), 2.0) -0.50 * pow(((ybar_p_rcvd-ybar_e)/sig_y), 2.0);
          Lum = bParams->Lc * exp(arg)/(sig_x*sig_y);

          printf("Lc:%f\n", bParams->Lc);
          printf("sig_x_e:%f\n", sig_x_e);
          printf("sig_x_p:%f\n", sig_x_p_rcvd);
          printf("sig_y_e:%f\n", sig_y_e);
          printf("sig_y_p:%f\n", sig_y_p_rcvd);
           
          // cout<<"sig_y_e:"<<sig_y_e<<endl;
          // cout<<"sig_y_p:"<<sig_y_p_rcvd<<endl;
          printf("Lum:%f\n", Lum);

          Lum_total += Lum;
				
          if(world_rank != 0){
            MPI_Isend(&Lum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, 
                      &myRequest_send_Sumary[0]);
            MPI_Isend(x_p_mp, 
                      (bunch_count_of_gpu_p[world_rank])*(NCOL+2)*bParams->Npart_p ,
                      MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, 
                      &myRequest_send_Sumary[1]);
            MPI_Isend(x_e_mp, 
                      (bunch_count_of_gpu_e[world_rank])*(NCOL+2)*bParams->Npart_e , 
                      MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, 
                      &myRequest_send_Sumary[2]);
          }
        }
        
        if(world_rank == 0){
          //std::cout<<"Lum is "<<Lum<<"\n";
          memcpy(beam->x_p, x_p_mp,
                 sizeof(double)*(bunch_count_of_gpu_p[0])*(NCOL+2)*bParams->Npart_p);
          memcpy(beam->x_e, x_e_mp,
                 sizeof(double)*(bunch_count_of_gpu_e[0])*(NCOL+2)*bParams->Npart_e);
          for(int i = 1; i < world_size; i++) {
            if(final_e_bunch_of_gpu[i] != -1){
              MPI_Irecv(&Lum_rcvd, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, 
                        &myRequest_send_Sumary[0]);
              Lum_total += Lum_rcvd;
              MPI_Irecv(beam->x_p + starting_bunch_of_gpu_p[i]*Npart_p*(NCOL+2), 1,
                        MPI_DOUBLE, i, 0, MPI_COMM_WORLD, 
                        &myRequest_send_Sumary[1]);
              MPI_Irecv(beam->x_e + starting_bunch_of_gpu_e[i]*Npart_e*(NCOL+2), 1,
                        MPI_DOUBLE, i, 0, MPI_COMM_WORLD, 
                        &myRequest_send_Sumary[2]);
            }
          }
        }
				
        MPI_Barrier(MPI_COMM_WORLD);			
        temp_post_col = stop_timer_returntime(&timer_post_col, "Post Collide");
        //cout<<"comm time:"<<temp_post_col<<endl;
        POST_COLLIDE = temp_post_col + POST_COLLIDE;
        //if(world_rank ==0)
        {
          //cout<<"GPU_COLLIDE:"<<temp_col<<endl;
          //cout<<"POST_COLLIDE:"<<temp_post_col<<endl;
          //cout<<"POST_COLLIDE 2:"<<temp_post_col2<<endl;
          //cout<<"==================================="<<endl;
        }
        temp_col =0; 
        temp_post_col = 0;
        temp_post_col2 = 0;
      }
			
      lum_turn++;
      //cout<<"Lum Turn:"<<lum_turn<<endl;
      if(world_rank == 0)
        {			
          //io->dumpLum_mpi(lum_turn, Lum_total/num_bunch_p, "luminosity");
          //cout<<"Lum_turn "<<lum_turn<<endl;
          io->dumpLum(lum_turn, 
                      Lum_total/num_bunch_p, 
                      bParams->Lc, 
                      bParams->Lsl, 
                      bParams->N, 
                      xbar_e, 
                      ybar_e, 
                      xbar_p_rcvd, 
                      ybar_p_rcvd, 
                      sig_x_e, 
                      sig_y_e,
                      sig_x_p_rcvd, 
                      sig_y_p_rcvd, 
                      mom_x_e, 
                      mom_y_e, 
                      mom_x_p_rcvd, mom_y_p_rcvd,
                      pxbar_e, pybar_e, pzbar_e, sig_px_e, sig_py_e, sig_pz_e,
                      pxbar_p, pybar_p, pzbar_p, sig_px_p_rcvd, sig_py_p_rcvd, sig_pz_p, "luminosity", std::ios::app);
           

          // io->dumpParticles(bParams->ebunches, beam->x_e, 
          //                     bParams->Npart_e, 
          //                     NCOL, bParams->Nfreq, iTurn, "IC_e");
          // io->dumpParticles(bParams->pbunches, beam->x_p, 
          //                     bParams->Npart_p, 
          //                     NCOL, bParams->Nfreq, iTurn, "IC_p");
			
        }

      MPI_Barrier(MPI_COMM_WORLD);
      Lum_total = 0.0;
    }
  }
  //if(world_rank ==0)
  {
    //cout<<"TOTAL GPU_COLLIDE:"<<GPU_COLLIDE<<endl;
    //cout<<"TOTAL POST_COLLIDE:"<<POST_COLLIDE<<endl;
    //cout<<"TOTAL POST_COLLIDE2:"<<POST_COLLIDE2<<endl;
  }
	
  io->threadFinalize();
}



int* Simulator::generateMapData(Map *&map, int maxLen, int Npart, int Ncol){
  double *M = new double[maxLen * Ncol];
  int *it = new int[maxLen * Ncol * Ncol];
  for(int i = 0; i < maxLen; ++i){
    for(int j = 0; j < Ncol; ++j){
      M[j * maxLen + i] = map->M[i * 6 + j];
      for(int k = 0; k < 6; ++k)
	it[j * maxLen * 6 + i * 6 + k] = map->it[i * 6 * 6 + j * 6 + k];

    }
  }

  int *_it_ = new int[maxLen * Ncol * (Ncol + 2)];//additional 2 (8-B) for M  
  for(int j = 1; j <= maxLen; ++j){
    for(int i = 1; i <= Ncol; ++i){
      for(int k = 1; k <= Ncol; ++k){
	//it[(i - 1) * maxLen * Ncol + (j - 1) * Ncol + k - 1] = map->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k -1];
	_it_[(i - 1) * maxLen * (Ncol + 2) + (j - 1) * (Ncol + 2) + k - 1] = map->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k -1];
	//std::cout << map->it[(j - 1) * 6 * 6 + (i - 1) * 6 + k -1] << "\n";
      }
      int *tmp = (int *)&map->M[(j - 1) * Ncol + i - 1];
      _it_[(i - 1) * maxLen * (Ncol + 2) + (j - 1) * (Ncol + 2) + 6] = tmp[0];
      _it_[(i - 1) * maxLen * (Ncol + 2) + (j - 1) * (Ncol + 2) + 7] = tmp[1];
      //std::cout << map->M[(j - 1) * Ncol + i - 1] << "\n";
    }
  }

  delete it,M;
  return _it_;
}


bool Simulator::listen(int master_pid){
  bool listenStatus = false;
  char buff[BUFSIZE];
  MPI_Status stat;
  MPI_Recv(buff, BUFSIZE, MPI_CHAR, master_pid, TAG, MPI_COMM_WORLD, &stat);
  std::stringstream ss;
  double start = MPI_Wtime();
  if(strncmp(buff, MPI_DATA[EXIT].c_str(), MPI_DATA[EXIT].length()) == 0){
    std::cout<<"In if loop of listern"<<"\n";
    //Exit
    listenStatus = false;
  }else if(strncmp(buff, MPI_DATA[TRACKING_ONLY].c_str(), MPI_DATA[TRACKING_ONLY].length()) == 0){
    MPI_Status status;
    metadata meta_data;
    MPI_Datatype metadata_type = getMetadataType();
    MPI_Recv(&meta_data, 1, metadata_type, master_pid, META_TAG, MPI_COMM_WORLD, &status);


    
    int *h_eqnsitM = new int[meta_data.maxLen_eqns * meta_data.Ncol * (meta_data.Ncol + 2 )];
    int *h_itM = new int[meta_data.maxLen * meta_data.Ncol * (meta_data.Ncol + 2 )];
    int *h_eqnsNrow = new int[meta_data.Ncol];
    int *h_Nrow = new int[meta_data.Ncol];
    
    MPI_Recv(h_eqnsitM, meta_data.maxLen_eqns * meta_data.Ncol * (meta_data.Ncol + 2 ), MPI_INT, master_pid, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(h_itM, meta_data.maxLen * meta_data.Ncol * (meta_data.Ncol + 2 ), MPI_INT, master_pid, TAG, MPI_COMM_WORLD, &status);

    MPI_Recv(h_eqnsNrow, meta_data.Ncol, MPI_INT, master_pid, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(h_Nrow, meta_data.Ncol, MPI_INT, master_pid, TAG, MPI_COMM_WORLD, &status);



    //Receive particle info
    // Probe for an incoming message from process zero
    MPI_Probe(master_pid, TAG, MPI_COMM_WORLD, &status);

    int size = 0;
    MPI_Get_count(&status, MPI_DOUBLE, &size);
    
    double *h_x = new double[size];
    MPI_Recv(h_x, size, MPI_DOUBLE, master_pid, TAG, MPI_COMM_WORLD, &status);

    double *d_x;
    int *d_Nrow = 0, *d_itM = 0;
    int *d_eqnsNrow = 0, *d_eqnsitM = 0;

    int Npart = size/meta_data.Ncol;
    initDeviceMemory(d_eqnsitM, h_eqnsitM,
		     d_eqnsNrow, h_eqnsNrow,
		     d_x, h_x,
		     meta_data.maxLen_eqns, Npart, meta_data.Ncol);
    
    initDeviceMemory(d_itM, h_itM,
		     d_Nrow, h_Nrow,
		     d_x, h_x,
		     meta_data.maxLen, Npart, meta_data.Ncol);

    double *dOpx = 0;
    QuadDebug(cudaMalloc((void **)&dOpx, sizeof(double) * Npart * meta_data.Ncol));
    int *dOutOfBound = 0;
    QuadDebug(cudaMalloc((void **)&dOutOfBound, sizeof(int) * Npart));
    thrust::device_ptr<int> dev_ptr(dOutOfBound);    
    thrust::fill(dev_ptr, dev_ptr + Npart, (int) 0);
  
    //quad::timer::event_pair timer_node;
    //quad::timer::start_timer(&timer_node);
    
    double time = 0;
    for(int iTurn = 1; iTurn <= bParams->Niter; ++iTurn){      
      double exec_time = applyMapGPU(dOutOfBound,
				     d_itM, d_eqnsitM, 
				     d_x, dOpx,
				     h_Nrow, h_eqnsNrow, 
				     h_x, 
				     meta_data.maxLen, meta_data.maxLen_eqns, 
				     Npart, meta_data.Ncol, bParams, iTurn);
      time += exec_time;
      
      dumpBeamByThread(this, d_x, h_x, Npart, meta_data.Ncol, iTurn, "dump.ebeam", std::ios::app, 1);
    }
    io->threadFinalize();
          
    //quad::timer::stop_timer(&timer_node, "GPU Tracking");

    ss.str("");
    ss << "Tracking took " << time/bParams->Niter << " ms per turn  in " << hostname << " (Rank = " << rank << ")\n" ;
    std::cout << ss.str();

    //write to rank file
    //io->dumpParticles(h_x, Npart, NCOL, bParams->Nfreq, bParams->Niter, "dump.ebeam");
    cudaFree(dOpx);

    listenStatus = true;
  }
  return listenStatus;
}

MPI_Datatype Simulator::getMetadataType(){
  MPI_Datatype metadata_type;
  
  const int num_ele = 1;
  int blocklens[num_ele];
  MPI_Aint disp[num_ele];
  MPI_Datatype types[num_ele];

  blocklens[0] = 4;
  disp[0] = 0;
  types[0] = MPI_INT;

  MPI_Type_create_struct( num_ele, blocklens, disp, types, &metadata_type );
  //MPI_Type_struct( num_ele, blocklens, disp, types, &metadata_type );
  MPI_Type_commit(&metadata_type);
  return metadata_type;
}


void Simulator::sendMetadata(int maxLen_eqns, int maxLen, int Npart, int Ncol, BeamParams *bParams){
  metadata data;
  data.maxLen_eqns = maxLen_eqns;data.maxLen = maxLen;data.Npart = Npart;data.Ncol = Ncol;
  
  MPI_Datatype metadata_type = getMetadataType();
  

  for(int node_id = 0; node_id < num_tasks; ++node_id){
    if(node_id != 0){
      MPI_Send(&data, 1, metadata_type, node_id, META_TAG, MPI_COMM_WORLD);
    }
  }
}

//demonstrates that GPUs run concurrently in a simple way
__global__ void ex(int *deviceRank)
{
	int x=-1;
	cudaGetDevice(&x);
	//printf("from ex:%i\n",x);
	for(int i=0; i< 2000; i++)
	{
		printf("%i hello from %i x=%i\n", i , *deviceRank, x);
	}
}

void Diagnostics(int MPIrank)
{
	//GPU diagnostics
	int num_devices=-1;
	cudaGetDeviceCount(&num_devices);
	int deviceRank=-1;
	cudaDeviceProp devProp;
	cudaGetDevice(&deviceRank);
	cudaGetDeviceProperties(&devProp, deviceRank);
	printf("Num Devices: %i\n", num_devices);
	printf("Device MPI RANK:%i GPU Rank:%i, name:%s\n", MPIrank, deviceRank, devProp.name);
	//printf("Max Block Size:%i\n", devProp.maxGridSize);
	//int *d_a;
	//cudaMalloc((void **)&d_a, sizeof(int));
	//cudaMemcpy(d_a, &deviceRank, sizeof(int), cudaMemcpyHostToDevice);
	
	//MPI_Barrier(MPI_COMM_WORLD);
	//ex<<<1,1>>>(d_a);
	//cudaFree(d_a);
	
}

int main(int argc, char **argv){
	
  quad::timer::event_pair timer_node;
  
  //if(argv[1] != NULL) can enable this for cases where numGPus != num_mpi processes
	//cout<<"Parameters:"<<argv[1]<<endl;
  	
  cudaSetDevice(0);
  int len = 0, rc = 0;
  std::stringstream ss;

  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    std::stringstream ss;
    ss << "Error starting MPI program. Terminating.\n";
    PrintlnPID(ss.str(), 0);
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  	
  Simulator sim;
  MPI_Comm_size(MPI_COMM_WORLD,&sim.num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&sim.rank);
  MPI_Get_processor_name(sim.hostname, &len);
  if(sim.rank ==0)
	quad::timer::start_timer(&timer_node);
  /*if(sim.rank ==0 || sim.rank ==1)
	  cudaSetDevice(0);
  else
	  cudaSetDevice(1);*/
  //cudaSetDevice(sim.rank % 4);
  //cout<<"GPU:"<<sim.rank%4<<endl;
/*  if(sim.rank == 0)
	cudaSetDevice(0);
  else if(sim.rank%2 == 0)
	cudaSetDevice(0);
  else
	cudaSetDevice(1);
*/
  
  std::map<std::string, int> nodeMap;
  map<std::string, int>::iterator it;
  for(int node_id = 0; node_id < sim.num_tasks; ++node_id)
  {
    char* buf = new char[MPI_MAX_PROCESSOR_NAME];
    memcpy(buf, sim.hostname, sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    MPI_Bcast(buf, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, node_id, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
	
    if(nodeMap.find(buf) == nodeMap.end())
      nodeMap[buf] = 0;
    else
      nodeMap[buf]++;
    if(node_id == sim.rank)
	{
      sim.deviceId = nodeMap[buf];
	  cout<<"Before QuadDebug Rank:"<<sim.rank<<endl;
	  cudaError_t t =QuadDebug(cudaSetDevice(sim.deviceId));
      cout<<"QuadDebug:"<<t<<endl;
	  cout <<"node id"<< node_id << " " <<"buf"<< buf << " " <<"nodemap[buf]"<< nodeMap[buf] << "\n";
    }
  }
  
  //quad::timer::event_pair timer_node;
  //quad::timer::start_timer(&timer_node);
  sim.io->ReadInput(sim.bParams);
  //quad::timer::stop_timer(&timer_node, "ReadInput");

  //CPU Execution & MPI == STOP
  if(!sim.bParams->isGPU && sim.num_tasks > 1){
    std::stringstream ss;
    ss << "Error: [CPU Execution does not run on MPI Cluster]\n";
    ss << "Usage: ./exec\n";
    std::cout << ss.str();
    MPI_Finalize();
    return 0;
  }
  
  if(sim.bParams->isGPU && sim.bParams->iTrackOnly == 1 && (sim.bParams->Npart_e % (ILP * sim.num_tasks)) != 0){
    std::stringstream ss;
    ss << "Error: GPU implementation requires #. of particles in the simulation should be a multiple of ILP * #. of GPUs  ";
    ss << ILP << " * " << sim.num_tasks << "\n";
    std::cout << ss.str();
    MPI_Finalize();
    return 0;
  }

  if(sim.bParams->iTrackOnly == 1 && ILP !=4  && sim.bParams->isSymTr){
    std::stringstream ss;
    ss << "Error: Symplectic Tracking is not implemented with ILP = " << ILP << "\n";
    std::cout << ss.str();
    MPI_Finalize();
    return 0;
  }
  

  if(sim.rank || sim.rank == 0)
  {
    IO::FileCreatePID(0);
	Diagnostics(sim.rank);
	sim.Simulate(argc, argv);
/*    char buff[BUFSIZE];
    for(int node_id = 0; node_id < sim.num_tasks; ++node_id){
      if(node_id != 0){
        std::cout<<"Sending msg"<<"\n";
	strncpy(buff, MPI_DATA[EXIT].c_str(), BUFSIZE);
	MPI_Send(buff, BUFSIZE, MPI_CHAR, node_id, TAG, MPI_COMM_WORLD);
      }
    }*/  
	
	
  }
  else
  {

    while(true){
      bool active = sim.listen();
      if(!active)break;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(sim.rank ==0)
	quad::timer::stop_timer(&timer_node, "Total Time");
  std::cout<<"Done simulation"<<"\n";
  if(sim.rank == sim.num_tasks-1){
    //quad::timer::start_timer(&timer_node);
    //IO::mergeDumps(sim.bParams, sim.num_tasks);
    //quad::timer::stop_timer(&timer_node, "Post-Processing");
  }
  MPI_Finalize();
  return 0;
}
