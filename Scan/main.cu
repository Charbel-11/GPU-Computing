#include "common.h"
#include "timer.h"

const double eps = 0.00001;
void checkIfEqual(double* cpuArray, double* gpuArray, unsigned int N){
	for(unsigned int i = 0; i < N; i++) {
        double diff = (cpuArray[i] - gpuArray[i])/cpuArray[i];	//division is to get relative error
        if(diff > eps || diff < -eps) {
            printf("Arrays are not equal (cpuArray[%u] = %e, GPUArray[%u] = %e)\n", i, cpuArray[i], i, gpuArray[i]);
            exit(0);
        }
    }
}

void scanCPU(double* input, double* output, unsigned int N) {
    output[0] = input[0];
    for(unsigned int i = 1; i < N; i++) {
        output[i] = f(output[i - 1], input[i]);
    }
}

// An inclusive scan finds the partial function of an array A (we get [A[0], f(A[0], A[1]), f(A[0], A[1], A[2]), ...]
// type 1 uses Kogge-Stone Scan while type 2 uses Brent-Kung Scan (both inclusive)
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int N = (argc > 2) ? (atoi(argv[2])) : 16000000;

	if (type == 1){ printf("Running Kogge-Stone Scan\n"); }
	else { printf("Running Brent-Kung Scan\n"); }

    double* input = (double*) malloc(N*sizeof(double));
    double* outputCPU = (double*) malloc(N*sizeof(double));
    double* outputGPU = (double*) malloc(N*sizeof(double));
    for(unsigned int i = 0; i < N; ++i)
        input[i] = 1.0*rand()/RAND_MAX;

    // Compute on CPU
    startTime(&timer);
    scanCPU(input, outputCPU, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

    // Compute on GPU
    startTime(&timer);
    scanGPU(input, outputGPU, N, type);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);

    // Verify result
    checkIfEqual(outputCPU, outputGPU, N);

    // Free memory
    free(input); 
    free(outputCPU); free(outputGPU);

    return 0;
}
