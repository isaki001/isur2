#ifndef COLLIDE_H
#define COLLIDE_H

#include "../common.h"
//#include "../util/cudaDebugUtil.h"
//#include "../util/cudaMemoryUtil.h"
#include "../util/cudaArchUtil.h"
#include "../util/cudaTimerUtil.h"
#include "../util/Util.h"


class Collide{

 public:
void collide(int, int *, int *, double *, double *, BeamParams *, int, int, int, int, int, int);

void collide_back(int *, int *, double *, double *, BeamParams *, int, int, int, int, int, int);


};

#endif /* COLLIDE_H */
