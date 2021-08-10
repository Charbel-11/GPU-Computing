#include "reduction.h"
#include "../Helper_Code/timer.h"

#define BLOCK_DIM 1024  
#define COARSE_FACTOR 3

template <typename T>
__global__ void reduceKernel(const T* input, T* partialSums, unsigned int N) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    // Loading data to shared memory
    __shared__ T input_s[BLOCK_DIM];
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
template <typename T>
__global__ void reduceKernelWithThreadCoarsening(const T* input, T* partialSums, unsigned int N) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x * COARSE_FACTOR;
    unsigned int i = segment + threadIdx.x;

    // Loading data to shared memory
    __shared__ T input_s[BLOCK_DIM];
	T sum = identity;
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

template <typename T>
T reduceGPUHelper(const T* input_d, unsigned int N, unsigned int type){
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
	if (type != 1) { numElementsPerBlock *= COARSE_FACTOR; }
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;

    T* partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(T));
    
	if (type == 1){ reduceKernel<T> <<< numBlocks, numThreadsPerBlock >>> (input_d, partialSums_d, N); }
	else { reduceKernelWithThreadCoarsening<T> <<< numBlocks, numThreadsPerBlock >>> (input_d, partialSums_d, N); }

    T sum;
    if (numBlocks > 1){ sum = reduceGPUHelper<T>(partialSums_d, numBlocks, type); }
    else{ cudaMemcpy(&sum, partialSums_d, sizeof(T), cudaMemcpyDeviceToHost); }

    cudaFree(partialSums_d);
    return sum;
}

template <typename T>
T reduceGPU(const T* input, unsigned int N, unsigned int type) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    T *input_d;
    cudaMalloc((void**) &input_d, N*sizeof(T));
	cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
	cudaMemcpy(input_d, input, N*sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    // Calling kernel
    startTime(&timer);
    T sum = reduceGPUHelper<T>(input_d, N, type);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

    // Freeing memory
    startTime(&timer);
    cudaFree(input_d); 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    return sum;
}

template double reduceGPU(const double* input, unsigned int N, unsigned int type);