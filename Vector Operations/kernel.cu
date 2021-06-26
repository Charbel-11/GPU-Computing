#include "common.h"
#include "timer.h"

/*
 * Macro to avoid duplication of error checking code for functions
 * such as cudaMalloc(), cudaMemcpy() and cudaFree()
 */
#define cudaErrorCheck(error) { gpuAssert((error), __FILE__, __LINE__); }

/*
 * Abort is set to True by default in order to immediately stop program execution
 */ 
void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true) {
   
    if (code != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s in file: %s (line: %d)\n", cudaGetErrorString(code), file, line);
      if (abort) {
          exit(code);
      }
   }

}

__global__ void vectorAdditionKernel(double* a, double* b, double* c, unsigned int N) {
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (i < N){
		c[i] = a[i] + b[i];
    }

}

void vectorAdditionGPU(double* a, double* b, double* c, unsigned int M) {

    Timer timer;

    /*
     * Allocate GPU memory
     */
    startTime(&timer);

    double *a_d, *b_d, *c_d;
    cudaError_t errMallocA = cudaMalloc((void **) &a_d, M * sizeof(double));
    cudaErrorCheck(errMallocA);

    cudaError_t errMallocB = cudaMalloc((void **) &b_d, M * sizeof(double));
    cudaErrorCheck(errMallocB);

    cudaError_t errMallocC = cudaMalloc((void **) &c_d, M * sizeof(double));
    cudaErrorCheck(errMallocC);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    /*
     * Copy data to GPU from Host
     */
    startTime(&timer);

    cudaError_t errMemcpyA = cudaMemcpy(a_d, a, M * sizeof(double), cudaMemcpyHostToDevice);
    cudaErrorCheck(errMemcpyA);

    cudaError_t errMemcpyB = cudaMemcpy(b_d, b, M * sizeof(double), cudaMemcpyHostToDevice);
    cudaErrorCheck(errMemcpyB);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");
    
    /* 
     * Call kernel
     */
    startTime(&timer);

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (M + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vecMax_kernel<<< numBlocks, numThreadsPerBlock >>>(a_d, b_d, c_d, M);

    /*
     * Call cudaGetLastError() first in order to check for any argument errors in the kernel
     *  
     * Call error-checking macro on cudaDeviceSynchronize() afterwards in order to wait for the
     * kernel to completely finish and check for any error while executing the kernel code
     */ 
    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaDeviceSynchronize());
    
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);
    
    /*
     * Copy data from GPU to Host
     */
    startTime(&timer);

    cudaError_t errMemcpyC = cudaMemcpy(c, c_d, M * sizeof(double), cudaMemcpyDeviceToHost);
    cudaErrorCheck(errMemcpyC);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");
    
    /*
     * Free GPU memory
     */
    startTime(&timer);

    cudaError_t errFreeA = cudaFree(a_d);
    cudaErrorCheck(errFreeA);

    cudaError_t errFreeB = cudaFree(b_d);
    cudaErrorCheck(errFreeB);

    cudaError_t errFreeC = cudaFree(c_d);
    cudaErrorCheck(errFreeC);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}