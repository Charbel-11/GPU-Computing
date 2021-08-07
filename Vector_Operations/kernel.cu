#include "common.h"
#include "timer.h"

#define cudaErrorCheck(error) { gpuAssert((error), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, const int line) {
    if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s in file %s at line %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

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
    cudaError_t errMallocA = cudaMalloc((void **) &a_d, N * sizeof(double)); cudaErrorCheck(errMallocA);
    cudaError_t errMallocB = cudaMalloc((void **) &b_d, N * sizeof(double)); cudaErrorCheck(errMallocB);
    cudaError_t errMallocC = cudaMalloc((void **) &c_d, N * sizeof(double)); cudaErrorCheck(errMallocC);

    cudaDeviceSynchronize();		//To get the correct time since GPU/CPU run asynchronously
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);

    cudaError_t errMemcpyA = cudaMemcpy(a_d, a, N * sizeof(double), cudaMemcpyHostToDevice); cudaErrorCheck(errMemcpyA);
    cudaError_t errMemcpyB = cudaMemcpy(b_d, b, N * sizeof(double), cudaMemcpyHostToDevice); cudaErrorCheck(errMemcpyB);

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
    cudaErrorCheck(cudaGetLastError());			//For arguments errors
    cudaErrorCheck(cudaDeviceSynchronize());	//For execution error in the kernel
    
    stopTime(&timer);
    printElapsedTime(timer, "Running the kernel time", GREEN);
    
    //Copying data from GPU to Host
    startTime(&timer);

    cudaError_t errMemcpyC = cudaMemcpy(c, c_d, N * sizeof(double), cudaMemcpyDeviceToHost);  cudaErrorCheck(errMemcpyC);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");
    
    //Freeing GPU memory
    startTime(&timer);

    cudaError_t errFreeA = cudaFree(a_d); cudaErrorCheck(errFreeA);
    cudaError_t errFreeB = cudaFree(b_d); cudaErrorCheck(errFreeB);
    cudaError_t errFreeC = cudaFree(c_d); cudaErrorCheck(errFreeC);
    
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Deallocation time");
}
