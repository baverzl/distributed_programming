#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cstdio>

using namespace std;

#define SIZE 1024 * 1024
const int N = 1024;

float h_A[SIZE];
float h_B[SIZE];
float h_C[SIZE];

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (row < N && col < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) 
            tmpSum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = tmpSum;
}


void matrixMultiplication(float *A, float *B, float *C, int N){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 1024 threads per block
    dim3 threadsPerBlock(1, 1);
    dim3 blocksPerGrid(1024, 1024);

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

int main(void)
{
	
	float *d_A, *d_B, *d_C;

	for(int i = 0; i < N; i++) {
			h_A[i * N + i] = 1;
			h_B[i * N + i] = 2;
	}
	
	cudaMalloc((void **) &d_A, SIZE * sizeof(float));
	cudaMalloc((void **) &d_B, SIZE * sizeof(float));
	cudaMalloc((void **) &d_C, SIZE * sizeof(float));

	cudaMemcpy(d_A, h_A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	matrixMultiplication(d_A, d_B, d_C, N);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			cout << h_C[i * N + j] << " ";
		}
		cout << endl;
	}
	
	printf("Time for the kernel: %fms\n", time);

	return 0;
}

