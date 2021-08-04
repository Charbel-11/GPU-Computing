#include "common.h"
#include "timer.h"

#define COARSE_FACTOR 4

__global__ void histogramKernelBasic(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width*height) {
        unsigned char b = image[i];
        atomicAdd(&bins[b], 1);
    }
}

__global__ void histogramKernelPrivate(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int curIdx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) { bins_s[i] = 0; }
    __syncthreads();

    if (curIdx < width * height) {
        unsigned char b = image[curIdx];
        atomicAdd(&bins_s[b], 1);
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&bins[i], bins_s[i]);
    }
}

__global__ void histogramKernelPrivateCoarsed(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int curIdx = (blockIdx.x * blockDim.x * COARSE_FACTOR) + threadIdx.x;

    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) { bins_s[i] = 0; }
    __syncthreads();
    
    for(unsigned int c = 0; c < COARSE_FACTOR; c++) {
        unsigned int curPos = curIdx + c * blockDim.x;
        
        if (curPos < width * height) {
            unsigned char b = image[curPos];
            atomicAdd(&bins_s[b], 1);
        }
        else { break; }
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&bins[i], bins_s[i]);
    }
}

void histogramGPUHelper(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height, unsigned int type){
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numElements = width * height;
    unsigned int numElementsPerBlock = numThreadsPerBlock;

    if (type != 1 && type != 2){ numElementsPerBlock *= COARSE_FACTOR; }
    unsigned int numBlocks = (numElements + numElementsPerBlock - 1) / numElementsPerBlock;

    if (type == 1){ histogramKernelBasic <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height); }
    else if (type == 2){ histogramKernelPrivate <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height); }
    else { histogramKernelPrivateCoarsed <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height); }
}

void histogramGPU(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height, unsigned int type){
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    unsigned char *image_d; unsigned int *bins_d;
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &bins_d, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    // Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    // Computing on GPU
    startTime(&timer);
    histogramGPUHelper(image_d, bins_d, width, height, type);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

	// Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(bins, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

    // Freeing GPU memory
    startTime(&timer);
    cudaFree(image_d); cudaFree(bins_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}