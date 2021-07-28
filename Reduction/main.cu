#include "common.h"
#include "timer.h"

const float eps = 0.00001;
void checkIfEqual(float cpuVal, float gpuVal){
    float diff = (cpuVal - gpuVal)/cpuVal;	//division is to get relative error
    if(diff > eps || diff < -eps) {
        printf("Values are not equal (cpuVal = %e, gpuVal = %e)\n", cpuVal, gpuVal);
        exit(0);
    }
}

float reduceCPU(float* input, unsigned int N) {
    float sum = identity;
    for(unsigned int i = 0; i < N; ++i) {
        sum = f(sum, input[i]);
    }
    return sum;
}

//Reduces an array A into f(A[1],A[2],...,A[n])
// type 1: usual parallelized reduction;  type 2: uses thread coarsening
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
	unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int N = (argc > 2) ? (atoi(argv[2])) : 16000000;
	
	if (type == 1){ printf("Running parallelized reduction\n"); }
	else { printf("Running parallelized reduction with thread coarsening\n"); }
	
    float* input = (float*) malloc(N*sizeof(float));
    for (unsigned int i = 0; i < N; ++i)
        input[i] = 1.0*rand()/RAND_MAX;
    
    // Compute on CPU
    startTime(&timer);
    float cpuVal = reduceCPU(input, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

    // Compute on GPU
    startTime(&timer);
    float gpuVal = reduceGPU(input, N, type);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);

    // Verify result
	checkIfEqual(cpuVal, gpuVal);

    // Free memory
    free(input);

    return 0;
}

