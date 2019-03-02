#!/bin/sh
#time mpirun -np 9 -hostfile hosts ./5-2 < input.txt
#time mpirun -np 9 -hostfile hosts ./5-3 < input.txt 
time mpirun -np 1 -hostfile hosts ./5-2 < input.txt
#time mpirun -np 1 -hostfile hosts ./5-3 < input.txt
#mpirun -np 4 -hostfile hosts ./third
