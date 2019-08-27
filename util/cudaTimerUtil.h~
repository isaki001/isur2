#include<string>
namespace quad{
  namespace timer{
    struct event_pair{
      cudaEvent_t start;
      cudaEvent_t end;
    };


    inline void check_cuda_error(char *message){
      cudaThreadSynchronize();
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
	{
	  printf("CUDA error after %s: %s\n", message, cudaGetErrorString(error));
	}
    }


    inline void start_timer(event_pair * p){
      cudaEventCreate(&p->start);
      cudaEventCreate(&p->end);
      cudaEventRecord(p->start, 0);
    }


    inline void stop_timer(event_pair * p, std::string kernel_name, int pid = 0){
      cudaEventRecord(p->end, 0);
      cudaEventSynchronize(p->end);

      float elapsed_time;
      cudaEventElapsedTime(&elapsed_time, p->start, p->end);

      int rank = 0, len = 0;
      char hostname[MPI_MAX_PROCESSOR_NAME];

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Get_processor_name(hostname, &len);

      std::stringstream ss;
      ss << kernel_name << " took " << elapsed_time << " ms in " << hostname << " (Rank = " << rank << ")\n" ;
      std::cout << ss.str() << "\n";
      //PrintlnPID(ss.str(), pid);

      //printf("%s took %.1f ms\n",kernel_name, elapsed_time);
      cudaEventDestroy(p->start);
      cudaEventDestroy(p->end);
    }

    inline float stop_timer_returntime(event_pair * p, std::string kernel_name){
      cudaEventRecord(p->end, 0);
      cudaEventSynchronize(p->end);

      float elapsed_time;
      cudaEventElapsedTime(&elapsed_time, p->start, p->end);
      // printf("%s took %.1f ms\n",kernel_name, elapsed_time);
      cudaEventDestroy(p->start);
      cudaEventDestroy(p->end);
      return elapsed_time;
    }


  }

}
