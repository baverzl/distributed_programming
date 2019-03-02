#!/bin/sh

export GOMP_CPU_AFFINITY="0 1 2 3"
export OMP_NUM_THREADS=4
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res8.txt 

export GOMP_CPU_AFFINITY="0-1 2-3 4-5 6-7"
export OMP_NUM_THREADS=8
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res7.txt

export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
export OMP_NUM_THREADS=8
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res6.txt

export GOMP_CPU_AFFINITY="0 4 1 5 2 6 3 7"
export OMP_NUM_THREADS=8
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res5.txt

export GOMP_CPU_AFFINITY="0 1 2 3"
export OMP_NUM_THREADS=8
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res4.txt

export GOMP_CPU_AFFINITY="0 1"
export OMP_NUM_THREADS=8
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res3.txt

export GOMP_CPU_AFFINITY="0"
export OMP_NUM_THREADS=8
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res2.txt

export GOMP_CPU_AFFINITY="0-3 2-5 4-7 6-1"
export OMP_NUM_THREADS=32
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res1.txt

export GOMP_CPU_AFFINITY="0-3 4-7 2-5 6-1"
export OMP_NUM_THREADS=32
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res0.txt

export GOMP_CPU_AFFINITY="0-3 4-7 2-5 6-1"
export OMP_NUM_THREADS=16
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res0-1.txt

export GOMP_CPU_AFFINITY="0-3 4-7 2-5 6-1"
export OMP_NUM_THREADS=64
./mandelbrot 2>/dev/null | grep Time | awk -F" " '{print $3}' >> res0-2.txt





