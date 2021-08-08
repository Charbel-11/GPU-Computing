#include "sort.h"
#include "../Helper_Code/timer.h"

#define BLOCK_DIM 1024  
#define ELEMENTS_PER_MERGE_THREAD 16
#define THREADS_PER_MERGE_BLOCK 128
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
        if (i > 0 && j < m && A[i-1] > B[j]) { r = i; }
        else if (j > 0 && i < n && B[j-1] > A[i]){ l = i; }
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
__global__ void mergeSortKernel(const T* input_d, T* output_d, T* tempOutput_d, unsigned int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i < N) { output_d[i] = input_d[i]; }
    __syncthreads();
    
    bool outputIsCorrect = true;
    for (unsigned int stride = 1; stride < N; stride *= 2) {
        if (i < (N + 2 * stride - 1) / (2 * stride)){
            unsigned int startIdx = i * 2 * stride;
            if (startIdx + stride < N){
                unsigned int n = stride, m = stride;
                if (startIdx + 2 * stride - 1 >= N) { m = N - (startIdx + stride); }
                
                unsigned int numBlocks = (n+m + ELEMENTS_PER_MERGE_BLOCK - 1) / ELEMENTS_PER_MERGE_BLOCK;
                mergeKernel<T> <<<numBlocks, THREADS_PER_MERGE_BLOCK >>> (&output_d[startIdx], &output_d[startIdx + stride], tempOutput_d, n, m);
            }
        }
        __syncthreads();

        T* tmp = tempOutput_d;
        tempOutput_d = output_d;
        output_d = tmp;
        outputIsCorrect = !outputIsCorrect;
    }

    if (!outputIsCorrect && i < N){ output_d[i] = tempOutput_d[i]; }
}

template <typename T>
void mergeSortGPUHelper(const T* input_d, T* output_d, unsigned int N){
    unsigned int numBlocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;

    T *tempOutput_d;
    cudaMalloc((void**) &tempOutput_d, N*sizeof(T));

    mergeSortKernel<T> <<< numBlocks, BLOCK_DIM >>> (input_d, output_d, tempOutput_d, N);    

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