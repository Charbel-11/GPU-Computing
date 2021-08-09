#include "SpMV_ELL.h"
#include "../Helper_Code/timer.h"

#define BLOCK_DIM 1024

template <typename T>
__global__ void SpMV_ELL_Kernel(const ELLMatrix<T> ellMatrix, const T* inVector, T* outVector){
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ellMatrix.numRows){ return; }

    T sum = 0;
    for(unsigned int i = 0; i < ellMatrix.nonZeroPerRow[row]; i++){
        unsigned int idx = i * ellMatrix.numRows + row;
        unsigned int col = ellMatrix.colIdxs[idx];
        sum += inVector[col] * ellMatrix.values[idx];
    }
    outVector[row] = sum;
}

template <typename T>
void SpMV_ELL_GPU(const ELLMatrix<T>& ellMatrix, const T* inVector, T* outVector) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    ELLMatrix<T> ellMatrix_d(ellMatrix.numRows, ellMatrix.numCols, ellMatrix.numNonzeros, true);
    ellMatrix_d.maxNonzerosPerRow = ellMatrix.maxNonzerosPerRow;
    ellMatrix_d.allocateArrayMemory();
    T *inVector_d, *outVector_d;
    cudaMalloc((void**) &inVector_d, ellMatrix_d.numCols*sizeof(T)); 
    cudaMalloc((void**) &outVector_d, ellMatrix_d.numRows*sizeof(T)); 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(ellMatrix_d.nonZeroPerRow, ellMatrix.nonZeroPerRow, ellMatrix_d.numRows*sizeof(unsigned int), cudaMemcpyHostToDevice);   
    cudaMemcpy(ellMatrix_d.colIdxs, ellMatrix.colIdxs, ellMatrix_d.maxNonzerosPerRow*ellMatrix_d.numRows*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(ellMatrix_d.values, ellMatrix.values, ellMatrix_d.maxNonzerosPerRow*ellMatrix_d.numRows*sizeof(T), cudaMemcpyHostToDevice);   
    cudaMemcpy(inVector_d, inVector, ellMatrix_d.numCols*sizeof(T), cudaMemcpyHostToDevice); 
    cudaMemset(outVector_d, 0, ellMatrix_d.numRows*sizeof(T));  
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    //Calling kernel
    startTime(&timer);
    unsigned int numBlocks = (ellMatrix_d.numRows + BLOCK_DIM - 1) / BLOCK_DIM;
    SpMV_ELL_Kernel<T> <<< numBlocks, BLOCK_DIM >>> (ellMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);
	
	//Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(outVector, outVector_d, ellMatrix_d.numRows*sizeof(T), cudaMemcpyDeviceToHost);
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

template void SpMV_ELL_GPU(const ELLMatrix<float>& ellMatrix, const float* inVector, float* outVector);