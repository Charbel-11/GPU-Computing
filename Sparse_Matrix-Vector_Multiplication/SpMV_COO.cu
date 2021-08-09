#include "SpMV_COO.h"
#include "../Helper_Code/timer.h"

#define BLOCK_DIM 1024

template <typename T>
__global__ void SpMV_COO_Kernel(const COOMatrix<T> cooMatrix, const T* inVector, T* outVector){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cooMatrix.numNonzeros){ return; }

    unsigned int row = cooMatrix.rowIdxs[i], col = cooMatrix.colIdxs[i];
    T curVal = cooMatrix.values[i];
    atomicAdd(&outVector[row], inVector[col] * curVal);
}

template <typename T>
void SpMV_COO_GPU(const COOMatrix<T>& cooMatrix, const T* inVector, T* outVector) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    COOMatrix<T> cooMatrix_d(cooMatrix.numRows, cooMatrix.numCols, cooMatrix.numNonzeros, true);
    cooMatrix_d.allocateArrayMemory();
    T *inVector_d, *outVector_d;
    cudaMalloc((void**) &inVector_d, cooMatrix_d.numCols*sizeof(T)); 
    cudaMalloc((void**) &outVector_d, cooMatrix_d.numRows*sizeof(T)); 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(cooMatrix_d.rowIdxs, cooMatrix.rowIdxs, cooMatrix_d.numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice); 
    cudaMemcpy(cooMatrix_d.colIdxs, cooMatrix.colIdxs, cooMatrix_d.numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooMatrix_d.values, cooMatrix.values, cooMatrix_d.numNonzeros*sizeof(T), cudaMemcpyHostToDevice);   
    cudaMemcpy(inVector_d, inVector, cooMatrix_d.numCols*sizeof(T), cudaMemcpyHostToDevice);   
    cudaMemset(outVector_d, 0, cooMatrix_d.numRows*sizeof(T));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    //Calling kernel
    startTime(&timer);
    unsigned int numBlocks = (cooMatrix_d.numNonzeros + BLOCK_DIM - 1) / BLOCK_DIM;
    SpMV_COO_Kernel<T> <<< numBlocks, BLOCK_DIM >>> (cooMatrix_d, inVector_d, outVector_d);    
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);
	
	//Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(outVector, outVector_d, cooMatrix_d.numRows*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

	//Freeing GPU memory
    startTime(&timer);
    cudaFree(inVector_d); cudaFree(outVector_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Deallocation time");
}

template void SpMV_COO_GPU(const COOMatrix<float>& cooMatrix, const float* inVector, float* outVector);