#include "convolution.h"
#include "../Helper_Code/timer.h"

#define OUT_BLOCK_DIM 32
#define IN_TILE_DIM 64
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(MASK_RADIUS))

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolutionKernel(float* input, float* output, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for(int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
            for(int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;
                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    sum += input[inRow*width + inCol] * mask_c[maskRow][maskCol];
                }
            }
        }
        output[outRow*width + outCol] = sum;
    }
}

__global__ void convolutionKernelWithTiling(float* input, float* output, unsigned int width, unsigned int height) {
    int curRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y - MASK_RADIUS;
    int curCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x - MASK_RADIUS;

    __shared__ float input_s[IN_TILE_DIM][IN_TILE_DIM];
    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
        input_s[threadIdx.y][threadIdx.x] = input[curRow*width + curCol];
    } 
	else {
        input_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Only threads in the output tile compute the sum
    if (threadIdx.x >= MASK_RADIUS && threadIdx.x < IN_TILE_DIM - MASK_RADIUS && threadIdx.y >= MASK_RADIUS && threadIdx.y < IN_TILE_DIM - MASK_RADIUS
	&& curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
        float sum = 0.0f;
        for(int maskRow = 0; maskRow < MASK_DIM; maskRow++)
            for(int maskCol = 0; maskCol < MASK_DIM; maskCol++)
                sum += input_s[maskRow + threadIdx.y - MASK_RADIUS][maskCol + threadIdx.x - MASK_RADIUS] * mask_c[maskRow][maskCol];

		output[curRow * width + curCol] = sum;   
	}
}

void convolutionGPU(float* input, float* output, float mask[MASK_DIM][MASK_DIM], unsigned int width, unsigned int height, unsigned int type) {
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    float *input_d, *output_d;
    cudaMalloc((void**) &input_d, width*height*sizeof(float)); 
    cudaMalloc((void**) &output_d, width*height*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    //Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(input_d, input, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM*MASK_DIM*sizeof(float)); //Copy mask to constant memory
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    //Calling kernel
    startTime(&timer);
    if (type == 1){
		dim3 numThreadsPerBlock(OUT_BLOCK_DIM, OUT_BLOCK_DIM);
		dim3 numBlocks((width + OUT_BLOCK_DIM - 1)/OUT_BLOCK_DIM, (height + OUT_BLOCK_DIM - 1)/OUT_BLOCK_DIM);
		convolutionKernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);
	}
	else{
		dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
		dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
		convolutionKernelWithTiling <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);
	}	
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);
	
	//Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(output, output_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

	//Freeing GPU memory
    startTime(&timer);
    cudaFree(input_d); cudaFree(output_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Deallocation time");
}
