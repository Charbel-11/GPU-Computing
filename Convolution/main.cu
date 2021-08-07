#include "common.h"
#include "../Helper_Code/timer.h"

const float eps = 0.00001;
void checkIfEqual(float* cpuArray, float* gpuArray, unsigned int N){
	for(unsigned int i = 0; i < N; i++) {
        float diff = (cpuArray[i] - gpuArray[i])/cpuArray[i];	//division is to get relative error
        if(diff > eps || diff < -eps) {
            printf("Arrays are not equal (cpuArray[%u] = %e, GPUArray[%u] = %e)\n", i, cpuArray[i], i, gpuArray[i]);
            exit(0);
        }
    }
}

void convolutionCPU(float mask[MASK_DIM][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height) {
    for (int outRow = 0; outRow < height; outRow++) {
        for (int outCol = 0; outCol < width; outCol++) {
            float sum = 0.0f;
            for(int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
                for(int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                    int inRow = outRow - MASK_RADIUS + maskRow;
                    int inCol = outCol - MASK_RADIUS + maskCol;
                    if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        sum += input[inRow*width + inCol] * mask[maskRow][maskCol];
                    }
                }
            }
            output[outRow*width + outCol] = sum;
        }
    }
}

// Maps an input matrix into an output matrix using convolution with mask
// type 1: uses one thread/output;  type 2: uses shared memory tiling
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    float mask[MASK_DIM][MASK_DIM];
	unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int height = (argc > 2) ? (atoi(argv[2])) : 5000;
    unsigned int width = (argc > 3) ? (atoi(argv[3])) : 5000;
	
	if (type == 1){ printf("Running basic parallelized Convolution\n"); }
	else { printf("Running parallelized Convolution with shared memory tiling\n"); }
	
    float* input = (float*) malloc(width*height*sizeof(float));
    float* output_cpu = (float*) malloc(width*height*sizeof(float));
    float* output_gpu = (float*) malloc(width*height*sizeof(float));
	
    for (unsigned int i = 0; i < MASK_DIM; i++)
        for (unsigned int j = 0; j < MASK_DIM; j++)
            mask[i][j] = (rand()/RAND_MAX)*100.0;
    for (unsigned int i = 0; i < height; i++)
        for (unsigned int j = 0; j < width; j++)
            input[i*width + j] = (rand()/RAND_MAX)*100.0;

    // Compute on CPU
    startTime(&timer);
    convolutionCPU(mask, input, output_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

	// Compute on GPU
    startTime(&timer);
    convolutionGPU(input, output_gpu, mask, width, height, type);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);

    // Verify result
	checkIfEqual(output_cpu, output_gpu, width*height);

    // Free memory
    free(input);
    free(output_cpu); free(output_gpu);

    return 0;
}

