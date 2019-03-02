#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLK_SIZE 100
#define MAX_NUM_THREADS_PER_BLK 1024

const int N = 1e2;

__global__ void reduce0(int *g_idata, int *g_odata) {
	__shared__ int sdata[MAX_NUM_THREADS_PER_BLK];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;

//	printf("%d %d %d\n", blockIdx.x, blockDim.x, tid);

	sdata[tid] = g_idata[i];
	__syncthreads();
	
	// do reduction in shared mem
	for(unsigned int s = 1; s < blockDim.x; s *= 2) {
		if(tid % (s * 2) == 0) { // Problem: highly divergent branching results in very poor performance
			sdata[tid] = (sdata[tid] < sdata[tid + s])? sdata[tid + s] : sdata[tid];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(void)
{
	int host_idata[N];
	int host_odata[MAX_NUM_THREADS_PER_BLK];
	int *dev_idata, *dev_odata;

	cudaMalloc((void **) &dev_idata, sizeof(int) * N);
	cudaMalloc((void **) &dev_odata, sizeof(int) * MAX_NUM_THREADS_PER_BLK);

	srand(time(NULL));

	for(int i = 0; i < N; i++) {
		host_idata[i] = rand() % 128;		
	}

	cudaMemcpy(dev_idata, host_idata, N * sizeof(int), cudaMemcpyHostToDevice);
	
	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	
	reduce0<<<BLK_SIZE, MAX_NUM_THREADS_PER_BLK>>> (dev_idata, dev_odata);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	printf("Time for the kernel: %f\n", time);
	
	cudaMemcpy(host_odata, dev_odata, BLK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	int max = -1;
	for(int i = 0; i < BLK_SIZE; i++)
		if(max < host_odata[i])
			max = host_odata[i];

	printf("reduction max : %d\n", max);

	return 0;

}

