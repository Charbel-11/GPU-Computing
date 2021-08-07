#include "common.h"
#include "timer.h"

#define BLOCK_DIM 32

#define cudaErrorCheck(error) { gpuAssert((error), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, const int line) {
    if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s in file %s at line %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

// A (MxK) * B (KxN) = C (MxN)
__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int outRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int outColumn = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (outRow < M && outColumn < N) {
        float sum = 0.0f;		// Register tiling
        for(unsigned int i = 0; i < K; i++) {
            sum += A[outRow*K + i] * B[i*N + outColumn];
        }
        C[outRow*N + outColumn] = sum;
    }
}

__global__ void matrixMultiplicationKernelWithTiling(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int outRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int outColumn = (blockIdx.x * blockDim.x) + threadIdx.x;
	float sum = 0.0f;		
	
	__shared__ float A_s[BLOCK_DIM][BLOCK_DIM];
	__shared__ float B_s[BLOCK_DIM][BLOCK_DIM];	
	
	for(unsigned int tileIdx = 0; tileIdx < (K + BLOCK_DIM - 1) / BLOCK_DIM; tileIdx++){
		if (outRow < M && (tileIdx*BLOCK_DIM + threadIdx.x) < K) {
			A_s[threadIdx.y][threadIdx.x] = A[outRow*K + (tileIdx*BLOCK_DIM + threadIdx.x)];
		}
		else { A_s[threadIdx.y][threadIdx.x] = 0.0; }
		
		if ((tileIdx*BLOCK_DIM + threadIdx.y) < K && outColumn < N) {
			B_s[threadIdx.y][threadIdx.x] = B[(tileIdx*BLOCK_DIM + threadIdx.y)*N + outColumn];
		}
		else { B_s[threadIdx.y][threadIdx.x] = 0.0; }
		
		__syncthreads();
		
		if (outRow < M && outColumn < N) {
			for(unsigned int i = 0; i < BLOCK_DIM; i++) {
				sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
			}
		}
		__syncthreads();
	} 
	
	if (outRow < M && outColumn < N) { C[outRow*N + outColumn] = sum; }
}

void matrixMultiplicationGPU(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K, unsigned int type) {
    Timer timer;

    //Allocating GPU memory
    startTime(&timer);

    float *A_d, *B_d, *C_d;
    cudaError_t errMallocA = cudaMalloc((void **) &A_d, M * K * sizeof(float)); cudaErrorCheck(errMallocA);
    cudaError_t errMallocB = cudaMalloc((void **) &B_d, K * N * sizeof(float)); cudaErrorCheck(errMallocB);
    cudaError_t errMallocC = cudaMalloc((void **) &C_d, M * N * sizeof(float)); cudaErrorCheck(errMallocC);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);

    cudaError_t errMemcpyA = cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice); cudaErrorCheck(errMemcpyA);
    cudaError_t errMemcpyB = cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice); cudaErrorCheck(errMemcpyB);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    //Calling kernel
    startTime(&timer);

    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (M + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    if (type == 1) { matrixMultiplicationKernel<<< numBlocks, numThreadsPerBlock >>>(A_d, B_d, C_d, M, N, K); }
	else { matrixMultiplicationKernelWithTiling<<< numBlocks, numThreadsPerBlock >>>(A_d, B_d, C_d, M, N, K); }
    cudaErrorCheck(cudaGetLastError());			//For arguments errors
    cudaErrorCheck(cudaDeviceSynchronize());	//For execution error in the kernel

    stopTime(&timer);
    printElapsedTime(timer, "Running the kernel time", GREEN);

    //Copying data from GPU to Host
    startTime(&timer);

    cudaError_t errMemcpyC = cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost); cudaErrorCheck(errMemcpyC);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

    //Freeing GPU memory
    startTime(&timer);

    cudaError_t errFreeA = cudaFree(A_d); cudaErrorCheck(errFreeA);
    cudaError_t errFreeB = cudaFree(B_d); cudaErrorCheck(errFreeB);
    cudaError_t errFreeC = cudaFree(C_d); cudaErrorCheck(errFreeC);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Deallocation time");
}

