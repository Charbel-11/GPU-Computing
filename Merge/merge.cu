#include "merge.h"
#include "../Helper_Code/timer.h"

#define ELEMENTS_PER_THREAD 16
#define THREADS_PER_BLOCK 128
#define ELEMENTS_PER_BLOCK (ELEMENTS_PER_THREAD * THREADS_PER_BLOCK)

template <typename T>
__host__ __device__ void mergeSequential(const T *A, const T *B, T *C, unsigned int n, unsigned int m){
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
    unsigned int k = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    if (k < m + n){ 
        unsigned int i = getCoRank<T>(A, B, n, m, k);
        unsigned int j = k-i;
        
        unsigned int kNext = (k + ELEMENTS_PER_THREAD < n + m) ? (k + ELEMENTS_PER_THREAD) : (n + m);
        unsigned int iNext = getCoRank<T>(A, B, n, m, kNext);
        unsigned int jNext = kNext - iNext;

        mergeSequential<T>(&A[i], &B[j], &C[k], iNext - i, jNext - j);
    }
}

template <typename T>
__global__ void mergeKernelTiled(const T *A, const T *B, T *C, unsigned int n, unsigned int m){
    unsigned int kBlock = blockIdx.x * ELEMENTS_PER_BLOCK;
    unsigned int kNextBlock = (blockIdx.x < gridDim.x - 1) ? (kBlock + ELEMENTS_PER_BLOCK) : (m + n);

    __shared__ unsigned int iBlock, iNextBlock;
    if (threadIdx.x == 0){
        iBlock = getCoRank<T>(A, B, n, m, kBlock);
        iNextBlock = getCoRank<T>(A, B, n, m, kNextBlock);
    }
    __syncthreads();

    unsigned int jBlock = kBlock - iBlock;
    unsigned int jNextBlock = kNextBlock - iNextBlock;

    __shared__ T A_s[ELEMENTS_PER_BLOCK];
    unsigned int nBlock = iNextBlock - iBlock;
    //Coalesced load into shared memory
    for(unsigned int i = threadIdx.x; i < nBlock; i += blockDim.x){
        A_s[i] = A[iBlock + i];
    }

    T* B_s = &A_s[nBlock];
    unsigned int mBlock = jNextBlock - jBlock;
    for(unsigned int j = threadIdx.x; j < mBlock; j += blockDim.x){
        B_s[j] = B[jBlock + j];
    }
    __syncthreads();

    // Merge in shared memory
    __shared__ T C_s[ELEMENTS_PER_BLOCK];
    unsigned int k = threadIdx.x * ELEMENTS_PER_THREAD;
    if (k < nBlock + mBlock) {
        unsigned int i = getCoRank<T>(A_s, B_s, nBlock, mBlock, k);
        unsigned int j = k-i;
        
        unsigned int kNext = (k + ELEMENTS_PER_THREAD < nBlock + mBlock) ? (k + ELEMENTS_PER_THREAD) : (nBlock + mBlock);
        unsigned int iNext = getCoRank<T>(A_s, B_s, nBlock, mBlock, kNext);
        unsigned int jNext = kNext - iNext;
    
        mergeSequential<T>(&A_s[i], &B_s[j], &C_s[k], iNext - i, jNext - j);        
    }
    __syncthreads();

    for(unsigned int k = threadIdx.x; k < nBlock + mBlock; k += blockDim.x){
        C[kBlock + k] = C_s[k];
    }
}

template <typename T>
void mergeCPU(const T *A, const T *B, T *C, unsigned int n, unsigned int m){
    mergeSequential<T>(A, B, C, n, m);
}

template <typename T>
void mergeGPUOnDevice(const T* A_d, const T *B_d, T *C_d, unsigned int n, unsigned int m, unsigned int type){
    unsigned int numBlocks = (n+m + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    if (type == 1) { mergeKernel<T> <<< numBlocks, THREADS_PER_BLOCK >>> (A_d, B_d, C_d, n, m); }
    else { mergeKernelTiled<T> <<< numBlocks, THREADS_PER_BLOCK >>> (A_d, B_d, C_d, n, m); }
}

template <typename T>
void mergeGPU(const T* A, const T *B, T *C, unsigned int n, unsigned int m, unsigned int type){
    Timer timer;

    // Allocating GPU memory
    startTime(&timer);
    T *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, n*sizeof(T));
    cudaMalloc((void**) &B_d, m*sizeof(T));
    cudaMalloc((void**) &C_d, (n+m)*sizeof(T));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    // Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(A_d, A, n*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, m*sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    // Computing on GPU
    startTime(&timer);
    mergeGPUOnDevice<T>(A_d, B_d, C_d, n, m, type);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);

	// Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(C, C_d, (n+m)*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

    // Freeing GPU memory
    startTime(&timer);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}

//Explicit instantiation to use different types
template void mergeCPU(const int *A, const int *B, int *C, unsigned int n, unsigned int m);
template void mergeGPU(const int *A, const int *B, int *C, unsigned int n, unsigned int m, unsigned int type);
template void mergeGPUOnDevice(const int* A_d, const int *B_d, int *C_d, unsigned int n, unsigned int m, unsigned int type);
template void mergeCPU(const unsigned int *A, const unsigned int *B, unsigned int *C, unsigned int n, unsigned int m);
template void mergeGPU(const unsigned int *A, const unsigned int *B, unsigned int *C, unsigned int n, unsigned int m, unsigned int type);
template void mergeGPUOnDevice(const unsigned int* A_d, const unsigned int *B_d, unsigned int *C_d, unsigned int n, unsigned int m, unsigned int type);