#include "sort.h"
#include "../Scan/scan.h"
#include "../Helper_Code/timer.h"
#include <algorithm>

#define BLOCK_DIM 1024  

__global__ void moveToDestKernel(const unsigned int* input, unsigned int* output, const unsigned int* prefOnes, unsigned int curBit, unsigned int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) { return; }

    if ((input[i] >> curBit) & 1) { output[i - prefOnes[i]] = input[i]; }
    else{ output[N - prefOnes[N] + prefOnes[i]] = input[i]; }
}

__global__ void extractOnesKernel(const unsigned int* input, unsigned int* ones, unsigned int curBit, unsigned int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N) { return; }

    if (i != N && ((input[i] >> curBit) & 1)) { ones[i] = 1; }
    else { ones[i] = 0; }
}

void radixSortGPUHelper(const unsigned int* input_d, unsigned int* output_d, unsigned int N){
    unsigned int numBlocks = ((N + 1) + BLOCK_DIM - 1) / BLOCK_DIM;

    unsigned int *prefOnes, *tempOutput;
    cudaMalloc((void**) &prefOnes, (N + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &tempOutput, N*sizeof(unsigned int));

    //First iteration done alone to move the values to the output array
    extractOnesKernel <<< numBlocks, BLOCK_DIM >>> (input_d, prefOnes, 0, N);
    scanGPUOnDevice<unsigned int>(prefOnes, prefOnes, N, 1, false);
    moveToDestKernel <<< numBlocks, BLOCK_DIM >>> (input_d, tempOutput, prefOnes, 0, N);
    std::swap(tempOutput, output_d);

    for(unsigned int b = 1; b < 32; b++){
        extractOnesKernel <<< numBlocks, BLOCK_DIM >>> (output_d, prefOnes, b, N);
        scanGPUOnDevice<unsigned int>(prefOnes, prefOnes, N, 1, false);
        moveToDestKernel <<< numBlocks, BLOCK_DIM >>> (output_d, tempOutput, prefOnes, b, N);    
        std::swap(tempOutput, output_d);
    }

    cudaFree(prefOnes); cudaFree(tempOutput);
}

void radixSortGPU(const unsigned int* input, unsigned int* output, unsigned int N){
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    unsigned int *input_d, *output_d;
    cudaMalloc((void**) &input_d, N*sizeof(unsigned int));
    cudaMalloc((void**) &output_d, N*sizeof(unsigned int));
	cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
	cudaMemcpy(input_d, input, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    // Calling kernel
    startTime(&timer);
	radixSortGPUHelper(input_d, output_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

	//Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(output, output_d, N*sizeof(unsigned int), cudaMemcpyDeviceToHost); 
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
