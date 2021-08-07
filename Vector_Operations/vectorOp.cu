#include "vectorOp.h"
#include "../Helper_Code/timer.h"

__global__ void vectorAdditionKernel(double* a, double* b, double* c, unsigned int N) {
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (i < N){ c[i] = a[i] + b[i]; }
}

__global__ void vectorMaxKernel(double* a, double* b, double* c, unsigned int N) {
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (i < N){ c[i] = (a[i] > b[i]) ? a[i] : b[i]; }
}

__global__ void vectorProductKernel(double* a, double* b, double* c, unsigned int N) {
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (i < N){ c[i] = a[i] * b[i]; }
}

void vectorOperationGPU(double* a, double* b, double* c, unsigned int N, unsigned int type) {
    Timer timer;

    //Allocating GPU memory
    startTime(&timer);
    double *a_d, *b_d, *c_d;
    cudaMalloc((void **) &a_d, N * sizeof(double));
    cudaMalloc((void **) &b_d, N * sizeof(double));
    cudaMalloc((void **) &c_d, N * sizeof(double));
    cudaDeviceSynchronize();		//To get the correct time since GPU/CPU run asynchronously
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(a_d, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");
    
    //Calling kernel
    startTime(&timer);
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    if (type == 1) { vectorAdditionKernel<<< numBlocks, numThreadsPerBlock >>>(a_d, b_d, c_d, N); }
    else if (type == 2) { vectorMaxKernel<<< numBlocks, numThreadsPerBlock >>>(a_d, b_d, c_d, N); }
    else if (type == 3) { vectorProductKernel<<< numBlocks, numThreadsPerBlock >>>(a_d, b_d, c_d, N); }
    stopTime(&timer);
    printElapsedTime(timer, "Running the kernel time", GREEN);
    
    //Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(c, c_d, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");
    
    //Freeing GPU memory
    startTime(&timer);
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Deallocation time");
}
