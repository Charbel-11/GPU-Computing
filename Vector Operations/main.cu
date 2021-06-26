#include "common.h"
#include "timer.h"

void checkIfEqual(double* c_cpu, double* c_gpu, unsigned int N){
	for(unsigned int i = 0; i < M; ++i) {
        double diff = (c_cpu[i] - c_gpu[i])/c_cpu[i];
        const double tolerance = 1e-9;
        if(diff > tolerance || diff < -tolerance) {
            printf("Mismatch at index %u (CPU result = %e, GPU result = %e)\n", i, c_cpu[i], c_gpu[i]);
            exit(0);
        }
    }
}

void vectorAdditionCPU(double* a, double* b, double* c, unsigned int N) {
    for(unsigned int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):32000000;
    double* a = (double*) malloc(M*sizeof(double));
    double* b = (double*) malloc(M*sizeof(double));
    double* c_cpu = (double*) malloc(M*sizeof(double));
    double* c_gpu = (double*) malloc(M*sizeof(double));
	
    for (unsigned int i = 0; i < M; ++i) {
        a[i] = rand(); b[i] = rand();
    }

    // Compute on CPU
    startTime(&timer);
    vectorAdditionCPU(a, b, c_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    vectorAdditionGPU(a, b, c_gpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    checkIfEqual(c_cpu, c_gpu, N);

    // Free memory
    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);

    return 0;

}

