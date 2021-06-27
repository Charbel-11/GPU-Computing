#include "common.h"
#include "timer.h"

const double eps = 1e-9;

void checkIfEqual(double* cpuArray, double* gpuArray, unsigned int N){
	for(unsigned int i = 0; i < N; i++) {
        double diff = (cpuArray[i] - gpuArray[i])/cpuArray[i];	//division is to get relative error
        if(diff > eps || diff < -eps) {
            printf("Arrays are not equal: (cpuArray[%u] = %e, GPUArray[%u] = %e)\n", i, cpuArray[i], i, gpuArray[i]);
            exit(0);
        }
    }
}

void vectorAdditionCPU(double* a, double* b, double* c, unsigned int N) {
    for(unsigned int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}
void vectorMaxCPU(double* a, double* b, double* c, unsigned int N) {
    for(unsigned int i = 0; i < N; i++) {
        c[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

void vectorOperationCPU(double* a, double* b, double* c, unsigned int N, unsigned int type){
	if (type == 1){ vectorAdditionCPU(a, b, c_cpu, N); }
	else if (type == 2){ vectorMaxCPU(a, b, c_cpu, N); }
}

//./main N type where N is the size of the vector and type is the operation (1 for add, 2 for max)
int main(int argc, char** argv) {
    cudaDeviceSynchronize();

    unsigned int N = (argc > 1) ? (atoi(argv[1])) : 32000000;
	unsigned int type = (argc > 2) ? (atoi(argv[2])) : 1;

    // Allocate memory and initialize data
    Timer timer;
    double* a = (double*) malloc(N*sizeof(double));
    double* b = (double*) malloc(N*sizeof(double));
    double* c_cpu = (double*) malloc(N*sizeof(double));
    double* c_gpu = (double*) malloc(N*sizeof(double));
	
	//Initializing two random arrays
    for (unsigned int i = 0; i < N; i++) { a[i] = rand(); b[i] = rand(); }

    // Compute on CPU
    startTime(&timer);
	vectorOperationCPU(a, b, c_gpu, N, type);	
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
	vectorOperationGPU(a, b, c_gpu, N, type);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    checkIfEqual(c_cpu, c_gpu, N);

    // Free memory
    free(a); free(b);
    free(c_cpu); free(c_gpu);

    return 0;
}
