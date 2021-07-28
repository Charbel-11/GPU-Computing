#include "common.h"
#include "timer.h"

#define BLOCK_DIM 1024  
#define COARSE_FACTOR 3

#define cudaErrorCheck(error) { gpuAssert((error), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, const int line) {
    if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s in file %s at line %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

__host__ __device__ double f(double a, double b){
	return a+b;
}

__global__ void reduceKernel(double* input, double* partialSums, unsigned int N) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    // Loading data to shared memory
    __shared__ double input_s[BLOCK_DIM];
    if (i + BLOCK_DIM < N) { input_s[threadIdx.x] = f(input[i], input[i + BLOCK_DIM]); }
    else if (i < N) { input_s[threadIdx.x] = input[i]; }
    else { input_s[threadIdx.x] = identity; }
    __syncthreads();

    // Reduction tree in shared memory with memory coalescing 
    for (unsigned int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride && threadIdx.x + stride < BLOCK_DIM) {
            input_s[threadIdx.x] = f(input_s[threadIdx.x], input_s[threadIdx.x + stride]);
        }      
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = input_s[0];
    }
}

// The higher the coarsening factor, the less we parallelize (which decreases the overhead for machines without enough resources)
__global__ void reduceKernelWithThreadCoarsening(double* input, double* partialSums, unsigned int N) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x * COARSE_FACTOR;
    unsigned int i = segment + threadIdx.x;

    // Loading data to shared memory
    __shared__ double input_s[BLOCK_DIM];
	double sum = identity;
	for(unsigned int tile = 0; tile < 2 * COARSE_FACTOR && i + tile*BLOCK_DIM < N; tile++){
		sum = f(sum, input[i+tile*BLOCK_DIM]);
	}
	input_s[threadIdx.x] = sum;
    __syncthreads();

    // Reduction tree in shared memory with memory coalescing 
    for (unsigned int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
            input_s[threadIdx.x] = f(input_s[threadIdx.x], input_s[threadIdx.x + stride]);
        }  
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = input_s[0];
    }
}

double reduceGPU(double* input, unsigned int N, unsigned int type) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
	
    double *input_d;
    cudaError_t errMallocA = cudaMalloc((void**) &input_d, N*sizeof(double)); cudaErrorCheck(errMallocA);
    
	cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
    
	cudaError_t errMemcpyA = cudaMemcpy(input_d, input, N*sizeof(double), cudaMemcpyHostToDevice); cudaErrorCheck(errMemcpyA);
	
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    // Allocating partial sums
    startTime(&timer);
	
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
	if (type != 1) { numElementsPerBlock *= COARSE_FACTOR; }
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
	
    double* partialSums = (double*) malloc(numBlocks*sizeof(double));
    double* partialSums_d;
    cudaError_t errMallocB = cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(double));  cudaErrorCheck(errMallocB);
	
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocating partial sums time");

    // Calling kernel
    startTime(&timer);
	if (type == 1){ reduceKernel <<< numBlocks, numThreadsPerBlock >>> (input_d, partialSums_d, N); }
	else { reduceKernelWithThreadCoarsening <<< numBlocks, numThreadsPerBlock >>> (input_d, partialSums_d, N); }
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

	//Copying data from GPU to Host
    startTime(&timer);

    cudaError_t errMemcpyB = cudaMemcpy(partialSums, partialSums_d, numBlocks*sizeof(double), cudaMemcpyDeviceToHost); cudaErrorCheck(errMemcpyB);
	
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

    // Reducing partial sums on CPU
    startTime(&timer);
	
    double sum = identity;
    for(unsigned int i = 0; i < numBlocks; ++i) {
        sum = f(sum, partialSums[i]);
    }
	
    stopTime(&timer);
    printElapsedTime(timer, "Reducing partial sums on host time");

    // Freeing memory
    startTime(&timer);
	
    cudaFree(input_d); cudaFree(partialSums_d);
    free(partialSums); 
	
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    return sum;
}

