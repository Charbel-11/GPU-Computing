#include "sort.h"
#include "../Merge/merge.h"
#include "../Helper_Code/timer.h"

#define BLOCK_DIM 1024  

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