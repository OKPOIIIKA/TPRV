#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <limits>

#define BLOCK_SIZE 32 //16

void getInfoCUDADevice(cudaDeviceProp& prop, int id) {
	printf("CUDA device %i name  - %s\n", id, prop.name);
	printf("CUDA device %i Warp size in threads  - %i\n", id, prop.warpSize);
	printf("CUDA device %i Maximum number of threads per block  - %i\n", id, prop.maxThreadsPerBlock);
	printf("CUDA device %i multiprocessors count  - %i\n", id, prop.multiProcessorCount);
	printf("CUDA device %i Maximum size of each dimension of a block  - %i %i %i\n", id, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("CUDA device %i Maximum size of each dimension of a grid  - %i %i %i\n", id, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

__global__ void matrixMult(const double *A, const double *B, double *result, int size) {
	//printf("blockIdx.y = %d,blockIdx.x = %d, threadIdx.y = %d, threadIdx.x = %d\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x);
    int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double sum = 0.0f;
	int ia = size * BLOCK_SIZE * by + size * ty;
	int ib = BLOCK_SIZE * bx + tx;
	int ic = size * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	for ( int k = 0; k < size; k++ )
		sum += A[ia + k] * B[ib + k*size];
	result[ic + size * ty + tx] = sum;
} 

void printResultMatr(const double* matr, int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			printf(" %f ", matr[i * size + j]);
		}
		printf("\n");
    }
}

bool is_equal(double x, double y) {
    return std::fabs(x - y) < 0.000001;
}

void compareMatrix(const double* f, const double* s, int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (!is_equal(f[i * size + j], s[i * size + j])) {
				printf("Matrixes not equal!\n");
				return;
			}
		}
	}
	printf("Matrixes is equal!\n");
}

double genDouble() {
    return sqrt((double)rand() / 1000000);
}

int main()
{
	int count;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&count);
	//printf("Count CUDA devices - %i\n", count);
	cudaGetDeviceProperties(&prop, count - 1);
	getInfoCUDADevice(prop, count - 1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	int size = 2048;
	for (int iter = 0; iter < 10; iter++) {
		printf("ex num: %d\n", iter);
		
		size_t byte_size = size * size * sizeof(double);
		double* h_A = (double*)malloc(byte_size);
		double* h_B = (double*)malloc(byte_size);
		double* h_C = (double*)malloc(byte_size);
		double* CPU_C = (double*)malloc(byte_size);

		for (int i = 0; i < size * size; ++i) {
			h_A[i] =  genDouble(); 
			h_B[i] =  genDouble(); 

			CPU_C[i] = 0;
		}
        // printf("\nh_A\n");
        // printResultMatr(h_A, size);
        // printf("\nh_B\n");
        // printResultMatr(h_B, size);
		// CPU (or host)
		
		printf("Scalar: \n");
		cudaEventRecord(start, 0);
		if (iter < 1) {
			for (int i = 0; i < size; ++i) {
				for (int j = 0; j < size; ++j) {
					for (int k = 0; k < size; ++k) {
						CPU_C[i * size + j] += h_A[i * size + k] * h_B[k * size + j];
					}
				}
			}
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float result_time_cpu;
		cudaEventElapsedTime(&result_time_cpu, start, stop);
		printf("Time: %f milliseconds\n", result_time_cpu);
		
		// printResultMatr(CPU_C, size);
		// GPU (or device)
		printf("GPU: \n");

		double* d_A = NULL;
		cudaMalloc((void**)&d_A, byte_size);
		cudaMemcpy(d_A, h_A, byte_size, cudaMemcpyHostToDevice);

		double* d_B = NULL;
		cudaMalloc((void**)&d_B, byte_size);
		cudaMemcpy(d_B, h_B, byte_size, cudaMemcpyHostToDevice);

		double* d_C = NULL;
		cudaMalloc((void**)&d_C, byte_size);

		cudaEventRecord(start, 0);

		const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		const dim3 grid(size / block.x, size / block.y);
		matrixMult <<< grid, block >>> (d_A, d_B, d_C, size);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float result_time_gpu;
		cudaEventElapsedTime(&result_time_gpu, start, stop);
		printf("Time: %f milliseconds\n", result_time_gpu);

		cudaMemcpy(h_C, d_C, byte_size, cudaMemcpyDeviceToHost);
        // printResultMatr(h_C, size);
		//compare
		compareMatrix(h_C, CPU_C, size);

		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		free(h_A);
		free(h_B);     
		free(h_C); 
		free(CPU_C);
	}
	cudaEventDestroy(start);  
	cudaEventDestroy(stop);

	return 0;
}