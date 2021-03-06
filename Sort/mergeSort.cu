#include "sort.h"
#include "../Helper_Code/timer.h"
#include <algorithm>

#define BLOCK_DIM 512  
#define ELEMENTS_PER_MERGE_THREAD 8
#define THREADS_PER_MERGE_BLOCK 64
#define ELEMENTS_PER_MERGE_BLOCK (ELEMENTS_PER_MERGE_THREAD * THREADS_PER_MERGE_BLOCK)

template <typename T>
__device__ void mergeSequential(const T *A, const T *B, T *C, unsigned int n, unsigned int m){
    unsigned int i = 0, j = 0, k = 0;
    while(i < n && j < m){
        if (A[i] < B[j]){ C[k++] = A[i++]; }
        else { C[k++] = B[j++]; }
    }
    while(i < n){ C[k++] = A[i++]; }
    while(j < m){ C[k++] = B[j++]; }
}

template <typename T>
__device__ unsigned int getCoRank(const T *A, const T *B, unsigned int n, unsigned int m, unsigned int k){
    unsigned int l = (k > m) ? (k - m) : 0;
    unsigned int r = (k < n) ? k : n;

    while(true){
        unsigned int i = (l + r) / 2;
        unsigned int j = k - i;
        if (i > 0 && j < m && A[i-1] > B[j]) { r = i - 1; }
        else if (j > 0 && i < n && B[j-1] > A[i]){ l = i + 1; }
        else { return i; }
    }
}

template <typename T>
__global__ void mergeKernel(const T *A, const T *B, T *C, unsigned int n, unsigned int m){
    unsigned int k = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_MERGE_THREAD;
    if (k < m + n){ 
        unsigned int i = getCoRank<T>(A, B, n, m, k);
        unsigned int j = k-i;
        
        unsigned int kNext = (k + ELEMENTS_PER_MERGE_THREAD < n + m) ? (k + ELEMENTS_PER_MERGE_THREAD) : (n + m);
        unsigned int iNext = getCoRank<T>(A, B, n, m, kNext);
        unsigned int jNext = kNext - iNext;

        mergeSequential<T>(&A[i], &B[j], &C[k], iNext - i, jNext - j);
    }
}

template <typename T>
__global__ void setOutputKernel(const T* input_d, T* output_d, unsigned int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;   
    if (i < N) { output_d[i] = input_d[i]; }
}

template <typename T>
__global__ void mergeSortKernel(T* output_d, T* tempOutput_d, unsigned int stride, unsigned int N, unsigned int numThreadsNeeded){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numThreadsNeeded){ return; }

    unsigned int startIdx = i * 2 * stride;
    unsigned int n = stride, m = stride;

    if (startIdx + stride >= N){  n = N - startIdx; m = 0; }
    else if (startIdx + 2 * stride - 1 >= N) { m = N - (startIdx + stride); }

    mergeSequential<T>(&output_d[startIdx], &output_d[startIdx + n], &tempOutput_d[startIdx], n, m);
}

template <typename T>
void mergeSortGPUHelper(const T* input_d, T* output_d, unsigned int N){
    unsigned int numBlocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    T *tempOutput_d;
    cudaMalloc((void**) &tempOutput_d, N*sizeof(T));

    setOutputKernel<T> <<< numBlocks, BLOCK_DIM >>> (input_d, output_d, N);
    bool outputIsCorrect = true;

    for (unsigned int stride = 1; stride < N; stride *= 2) {
        if (stride >= 10000){
            unsigned int numBlocks = (2*stride + ELEMENTS_PER_MERGE_BLOCK - 1) / ELEMENTS_PER_MERGE_BLOCK;
            for(unsigned int i = 0; i < N; i += 2 * stride){
                unsigned int n = stride, m = stride;
                if (i + stride >= N){ n = N - i; m = 0;}
                else if (i + 2*stride >= N) { m = N - (i+stride); }
                mergeKernel<T> <<< numBlocks, THREADS_PER_MERGE_BLOCK >>> (&output_d[i], &output_d[i+n], &tempOutput_d[i], n, m);
            }
        }
        else{
            unsigned int numThreadsNeeded = (N + 2 * stride - 1) / (2 * stride);
            numBlocks = (numThreadsNeeded + BLOCK_DIM - 1) / BLOCK_DIM;
            mergeSortKernel<T> <<< numBlocks, BLOCK_DIM >>> (output_d, tempOutput_d, stride, N, numThreadsNeeded);
        }
        
        std::swap(tempOutput_d, output_d);
        outputIsCorrect = !outputIsCorrect;
    }

    if (!outputIsCorrect){ 
        std::swap(tempOutput_d, output_d);
        numBlocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
        setOutputKernel<T> <<< numBlocks, BLOCK_DIM >>> (tempOutput_d, output_d, N);
    }

    cudaFree(tempOutput_d);
}

template <typename T>
void mergeSortGPU(const T* input, T* output, unsigned int N){
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    T *input_d, *output_d;
    cudaMalloc((void**) &input_d, N*sizeof(T));
    cudaMalloc((void**) &output_d, N*sizeof(T));
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
    mergeSortGPUHelper<T>(input_d, output_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

	//Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(output, output_d, N*sizeof(T), cudaMemcpyDeviceToHost); 
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

//Explicit instantiation to use different types
template void mergeSortGPU(const unsigned int* input, unsigned int* output, unsigned int N);