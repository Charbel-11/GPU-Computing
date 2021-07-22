#include "common.h"
#include "timer.h"

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

void matrixMultiplicationCPU(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    for (unsigned int row = 0; row < M; ++row) {
        for (unsigned int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for(unsigned int i = 0; i < K; ++i) {
                sum += A[row*K + i]*B[i*N + col];
            }
            C[row*N + col] = sum;
        }
    }
}

// Multiplies matrices A (MxK) and B (KxN) to get C (MxN)
// type 1: uses one thread/output;  type 2: uses memory tiling
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
	unsigned int M = (argc > 2) ? (atoi(argv[2])): 500;
    unsigned int K = (argc > 3) ? (atoi(argv[3])): 500;
    unsigned int N = (argc > 4) ? (atoi(argv[4])): 500;
	
	if (type == 1){ printf("Running basic parallelized Matrix Multiplication\n"); }
	else { printf("Running parallelized Matrix Multiplication with memory tiling\n"); }
		
    float* A = (float*) malloc(M*K*sizeof(float));
    float* B = (float*) malloc(K*N*sizeof(float));
    float* C_cpu = (float*) malloc(M*N*sizeof(float));
    float* C_gpu = (float*) malloc(M*N*sizeof(float));
	
    for (unsigned int i = 0; i < M; i++)
        for (unsigned int j = 0; j < K; j++)
            A[i*K + j] = 1.0*rand()/RAND_MAX;
    for (unsigned int i = 0; i < K; i++)
        for (unsigned int j = 0; j < N; j++)
            B[i*N + j] = 1.0*rand()/RAND_MAX;

    // Compute on CPU
    startTime(&timer);
    matrixMultiplicationCPU(A, B, C_cpu, M, N, K);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

    // Compute on GPU
    startTime(&timer);
    matrixMultiplicationGPU(A, B, C_gpu, M, N, K, type);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);

    // Verify result
	checkIfEqual(C_cpu, C_gpu, M*N);

    // Free memory
    free(A); free(B);
    free(C_cpu); free(C_gpu);

    return 0;
}

