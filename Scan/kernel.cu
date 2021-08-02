#include "common.h"
#include "timer.h"

#define BLOCK_DIM 1024

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

// Scans exactly one block
__global__ void scanKernelKoggeStone(double* input, double* output, double* partialSums, unsigned int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float buffer1_s[BLOCK_DIM], buffer2_s[BLOCK_DIM];
    float* prevBuffer_s = buffer1_s, *curBuffer_s = buffer2_s;
    
    if (i < N) { prevBuffer_s[threadIdx.x] = input[i]; }
    else {  prevBuffer_s[threadIdx.x] = identity; }
    __syncthreads();

    for(unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2){
        curBuffer_s[threadIdx.x] = prevBuffer_s[threadIdx.x] ;
        if (threadIdx.x >= stride){
            curBuffer_s[threadIdx.x] = f(curBuffer_s[threadIdx.x], prevBuffer_s[threadIdx.x - stride]);
        }
        __syncthreads();

        float *tmp = prevBuffer_s;
        prevBuffer_s = curBuffer_s;
        curBuffer_s = tmp;
    }

    if (threadIdx.x == BLOCK_DIM-1){
        partialSums[blockIdx.x] = prevBuffer_s[threadIdx.x];
    }

    if (i < N) { output[i] = prevBuffer_s[threadIdx.x]; }
}

__global__ void addKernelKoggeStone(double* output, double* partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x;

    if (blockIdx.x > 0 && segment + threadIdx.x < N) {
        output[segment + threadIdx.x] = f(output[segment + threadIdx.x], partialSums[blockIdx.x - 1]);
    }
}

__global__ void addKernelBrentKung(double* output, double* partialSums, unsigned int N) {
    unsigned int segment = 2 * blockIdx.x * blockDim.x;

    if (blockIdx.x > 0) {
        if (segment + threadIdx.x < N) {
            output[segment + threadIdx.x] = f(output[segment + threadIdx.x], partialSums[blockIdx.x - 1]);
        }
        if (segment + threadIdx.x + BLOCK_DIM < N) {
            output[segment + threadIdx.x + BLOCK_DIM] = f(output[segment + threadIdx.x + BLOCK_DIM], partialSums[blockIdx.x - 1]);
        }
    }
}

void scanHelperGPU(double* input_d, double* output_d, unsigned int N, unsigned int type) {
    Timer timer;

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock * ((type == 1) ? 1 : 2);
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    // Allocating partial sums
    double *partialSums_d;
    cudaError_t errMallocA = cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(double)); cudaErrorCheck(errMallocA);
    cudaDeviceSynchronize();

    // Calling the kernel to scan each block on its own
    startTime(&timer);
    if (type == 1) { scanKernelKoggeStone <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, partialSums_d, N); }
    else { } 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

    // Recursively scan partial sums then add
    if (numBlocks > 1) {
        scanHelperGPU(partialSums_d, partialSums_d, numBlocks, type);
        if (type == 1) { addKernelKoggeStone <<< numBlocks, numThreadsPerBlock >>> (output_d, partialSums_d, N); }
        else { addKernelBrentKung <<< numBlocks, numThreadsPerBlock >>> (output_d, partialSums_d, N); } 
    }

    // Free memory
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
}

void scanGPU(double* input, double* output, unsigned int N, unsigned int type) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    double *input_d, *output_d;
    cudaError_t errMallocA = cudaMalloc((void**) &input_d, N*sizeof(double)); cudaErrorCheck(errMallocA);
    cudaError_t errMallocB = cudaMalloc((void**) &output_d, N*sizeof(double)); cudaErrorCheck(errMallocB);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    // Copying data to GPU from Host
    startTime(&timer);
    cudaError_t errMemcpyA = cudaMemcpy(input_d, input, N*sizeof(double), cudaMemcpyHostToDevice);  cudaErrorCheck(errMemcpyA);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    // Computing on GPU
    scanHelperGPU(input_d, output_d, N, type);

	// Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(output, output_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

    // Freeing memory
    startTime(&timer);
    cudaFree(input_d); cudaFree(output_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}
