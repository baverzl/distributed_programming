#!/bin/sh
#echo "scan using MPIScan not dependent on n"
#mpirun -np 20 -hostfile hosts ./prefixsum_mpiscan_independent_of_np

echo "scan using MPIScan dependent on n"
mpirun -np 5 -hostfile hosts ./prefixsum_mpiscan
echo "scan using blocking calls"
mpirun -np 5 -hostfile hosts ./scan_blocking
echo "scan using non-blocking calls"
mpirun -np 5 -hostfile hosts ./scan_non_blocking
