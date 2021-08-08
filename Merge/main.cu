#include "merge.h"
#include "../Helper_Code/timer.h"

void checkIfEqual(int* arrayCPU, int* arrayGPU, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        if(arrayCPU[i] != arrayGPU[i]) {
            printf("Arrays are not equal (arrayCPU[%u] = %d, arrayGPU[%u] = %d)\n", i, arrayCPU[i], i, arrayGPU[i]);
            return;
        }
    }
}

// Merge function takes two sorted array and outputs one merged sorted array
// type 1 runs a basic parallelized algorithm, type 2 uses shared memory tiling
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int n = (argc > 2) ? (atoi(argv[2])) : 8000000;
    unsigned int m = (argc > 3) ? (atoi(argv[3])) : 8000000;

    if (type == 1){ printf("Running parallelized merge\n"); }
	else { printf("Running parallelized merge with shared memory tiling\n"); }

    int* A = (int*) malloc(n*sizeof(int));
    int* B = (int*) malloc(m*sizeof(int));
    int* outputCPU = (int*) malloc((n+m)*sizeof(int));
    int* outputGPU = (int*) malloc((n+m)*sizeof(int));
    for (unsigned int i = 0; i < n; i++) { A[i] = i; }
    for (unsigned int i = 0; i < m; i++) { B[i] = i; }

    // Compute on CPU
    startTime(&timer);
    mergeCPU<int>(A, B, outputCPU, n, m);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

    // Compute on GPU
    startTime(&timer);
    mergeGPU<int>(A, B, outputGPU, n, m, type);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);    

    // Verify result
    checkIfEqual(outputCPU, outputGPU, n + m);

    // Free memory
    free(A); free(B);
    free(outputCPU); free(outputGPU);

    return 0;
}
