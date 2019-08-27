#!/bin/bash
#n 5e-4 6e-4 7e-4 8e-4 9e-4 -5e-4 -6e-4 -7e-4 -8e-4 -9e-4
# 4e-4 2e-4 1e-4 9e-5 8e-5 7e-5 6e-5 5e-5 4e-5 3e-5 2e-5 1e-5 5e-6 1e-6 0.0 -1e-6 -5e-6 -1e-5 -2e-5 -3e-5 -4e-5 -5e-5 -6e-5 -7e-5 -8e-5 -9e-5 -1e-4 -2e-4 -3e-4 -4e-4  
#for x in {-9e-5..9e-5..(0.00001)}

for x in 1 2 3 5 10
do
    mkdir nSlice=$x
    #x = $x+20;
    cd nSlice=$x
    sed "s%<ns>%$x%g;" ../template/params.sim > params.sim
  
    cp ../template/launchBB3DScan.sh .
    cp ../template/qsubBB3DScan.sh .
    qsub job.sript
    cd ../
done    
