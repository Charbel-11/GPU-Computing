#include "scan.h"
#include "../Helper_Code/timer.h"

#define BLOCK_DIM 1024

__host__ __device__ double f(double a, double b){
	return a+b;
}

// Scans exactly one block
__global__ void scanKernelKoggeStone(double* input, double* output, double* partialSums, unsigned int N, bool inclusive){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float buffer1_s[BLOCK_DIM], buffer2_s[BLOCK_DIM];
    float* prevBuffer_s = buffer1_s, *curBuffer_s = buffer2_s;
    
    if (inclusive) {
        if (i < N) { prevBuffer_s[threadIdx.x] = input[i]; }
        else {  prevBuffer_s[threadIdx.x] = identity; }
    }
    else {
        if (threadIdx.x > 0 && i - 1 < N) { prevBuffer_s[threadIdx.x] = input[i - 1]; }
        else {  prevBuffer_s[threadIdx.x] = identity; }
    }
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
        if (!inclusive && i < N){ partialSums[blockIdx.x] = f(prevBuffer_s[threadIdx.x], input[i]); }
        else { partialSums[blockIdx.x] = prevBuffer_s[threadIdx.x]; }
    }

    if (i < N) { output[i] = prevBuffer_s[threadIdx.x]; }
}

// Scans exactly one block
__global__ void scanKernelBrentKung(double* input, double* output, double* partialSums, unsigned int N, bool inclusive){
    unsigned int segment = 2 * blockIdx.x * blockDim.x;
    unsigned int i1 = segment + threadIdx.x;
    unsigned int i2 = i1 + BLOCK_DIM;

    //We need to store this beforehand for exclusive scans as on recursive scan calls, input = output
    double toAdd = identity;
    if (threadIdx.x == 0 && !inclusive && 2 * BLOCK_DIM - 1 + segment < N){ toAdd = input[segment + 2 * BLOCK_DIM - 1]; }

    __shared__ double buffer_s[2 * BLOCK_DIM];

    if (inclusive){
        if (i1 < N) { buffer_s[threadIdx.x] = input[i1]; }
        else { buffer_s[threadIdx.x] = identity; }
        if (i2 < N) { buffer_s[threadIdx.x + BLOCK_DIM] = input[i2]; }
        else { buffer_s[threadIdx.x + BLOCK_DIM] = identity; }
    }
    else{
        if (threadIdx.x > 0 && i1 - 1 < N) { buffer_s[threadIdx.x] = input[i1 - 1]; }
        else { buffer_s[threadIdx.x] = identity; }
        if (i2 - 1 < N) { buffer_s[threadIdx.x + BLOCK_DIM] = input[i2 - 1]; }
        else { buffer_s[threadIdx.x + BLOCK_DIM] = identity; }
    }
    __syncthreads();

    // Reduction Step
    for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;  
        if (i < 2 * BLOCK_DIM) { buffer_s[i] = f(buffer_s[i], buffer_s[i - stride]); }
        __syncthreads();
    }

    // Post-reduction Step
    for (unsigned int stride = BLOCK_DIM/2; stride >= 1; stride /= 2) {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if (i + stride < 2 * BLOCK_DIM) { buffer_s[i + stride] = f(buffer_s[i + stride], buffer_s[i]); }
        __syncthreads();
    }

    // Store partial sum
    if (threadIdx.x == 0){
        if (!inclusive && 2 * BLOCK_DIM - 1 + segment < N){ 
            partialSums[blockIdx.x] = f(buffer_s[2 * BLOCK_DIM - 1], toAdd);
        }
        else { partialSums[blockIdx.x] = buffer_s[2 * BLOCK_DIM - 1]; }
    }

    // Store output
    if (i1 < N) { output[i1] = buffer_s[threadIdx.x]; } 
    if (i2 < N) { output[i2] = buffer_s[threadIdx.x + BLOCK_DIM]; }          
}

__global__ void addKernelKoggeStone(double* output, double* partialSums, unsigned int N, bool inclusive) {
    unsigned int segment = blockIdx.x * blockDim.x;

    if (blockIdx.x > 0 && segment + threadIdx.x < N) {
        output[segment + threadIdx.x] = f(output[segment + threadIdx.x], partialSums[blockIdx.x - inclusive]);
    }
}

__global__ void addKernelBrentKung(double* output, double* partialSums, unsigned int N, bool inclusive) {
    unsigned int segment = 2 * blockIdx.x * blockDim.x;

    if (blockIdx.x > 0) {
        if (segment + threadIdx.x < N) {
            output[segment + threadIdx.x] = f(output[segment + threadIdx.x], partialSums[blockIdx.x - inclusive]);
        }
        if (segment + threadIdx.x + BLOCK_DIM < N) {
            output[segment + threadIdx.x + BLOCK_DIM] = f(output[segment + threadIdx.x + BLOCK_DIM], partialSums[blockIdx.x - inclusive]);
        }
    }
}

void scanHelperGPU(double* input_d, double* output_d, unsigned int N, unsigned int type, bool inclusive) {
    Timer timer;

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock * ((type == 1) ? 1 : 2);
    const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    // Allocating partial sums
    double *partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(double));
    cudaDeviceSynchronize();

    // Calling the kernel to scan each block on its own
    startTime(&timer);
    if (type == 1) { scanKernelKoggeStone <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, partialSums_d, N, inclusive); }
    else { scanKernelBrentKung <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, partialSums_d, N, inclusive); } 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

    // Recursively scan partial sums then add
    if (numBlocks > 1) {
        scanHelperGPU(partialSums_d, partialSums_d, numBlocks, type, inclusive);
        if (type == 1) { addKernelKoggeStone <<< numBlocks, numThreadsPerBlock >>> (output_d, partialSums_d, N, inclusive); }
        else { addKernelBrentKung <<< numBlocks, numThreadsPerBlock >>> (output_d, partialSums_d, N, inclusive); } 
    }

    // Free memory
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
}

void scanGPU(double* input, double* output, unsigned int N, unsigned int type, bool inclusive) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    double *input_d, *output_d;
    cudaMalloc((void**) &input_d, N*sizeof(double));
    cudaMalloc((void**) &output_d, N*sizeof(double));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    // Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    // Computing on GPU
    scanHelperGPU(input_d, output_d, N, type, inclusive);

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
