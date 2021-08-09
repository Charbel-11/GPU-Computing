#include "SpMV_CSR.h"
#include "../Helper_Code/timer.h"

#define BLOCK_DIM 1024

template <typename T>
__global__ void SpMV_CSR_Kernel(const CSRMatrix<T> csrMatrix, const T* inVector, T* outVector){
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= csrMatrix.numRows){ return; }

    T sum = 0;
    for(unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; i++){
        unsigned int col = csrMatrix.colIdxs[i];
        sum += inVector[col] * csrMatrix.values[i];
    }
    outVector[row] = sum;
}

template <typename T>
void SpMV_CSR_GPU(const CSRMatrix<T>& csrMatrix, const T* inVector, T* outVector) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    CSRMatrix<T> csrMatrix_d(csrMatrix.numRows, csrMatrix.numCols, csrMatrix.numNonzeros, true);
    csrMatrix_d.allocateArrayMemory();
    T *inVector_d, *outVector_d;
    cudaMalloc((void**) &inVector_d, csrMatrix_d.numCols*sizeof(T)); 
    cudaMalloc((void**) &outVector_d, csrMatrix_d.numRows*sizeof(T)); 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(csrMatrix_d.rowPtrs, csrMatrix.rowPtrs, (csrMatrix_d.numRows + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice); 
    cudaMemcpy(csrMatrix_d.colIdxs, csrMatrix.colIdxs, csrMatrix_d.numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrMatrix_d.values, csrMatrix.values, csrMatrix_d.numNonzeros*sizeof(T), cudaMemcpyHostToDevice);   
    cudaMemcpy(inVector_d, inVector, csrMatrix_d.numCols*sizeof(T), cudaMemcpyHostToDevice); 
    cudaMemset(outVector_d, 0, csrMatrix_d.numRows*sizeof(T));  
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    //Calling kernel
    startTime(&timer);
    unsigned int numBlocks = (csrMatrix_d.numRows + BLOCK_DIM - 1) / BLOCK_DIM;
    SpMV_CSR_Kernel<T> <<< numBlocks, BLOCK_DIM >>> (csrMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);
	
	//Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(outVector, outVector_d, csrMatrix_d.numRows*sizeof(T), cudaMemcpyDeviceToHost);
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

template void SpMV_CSR_GPU(const CSRMatrix<float>& csrMatrix, const float* inVector, float* outVector);