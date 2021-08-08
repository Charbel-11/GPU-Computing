#include "sort.h"
#include "../Merge/merge.h"
#include "../Helper_Code/timer.h"
#include <algorithm>

bool checkIfSorted(const unsigned int* array, unsigned int N){
    for(int i = 1; i < N; i++){
        if (array[i] < array[i-1]){ return false; } 
    }
    return true;
}

void checkIfEqual(const unsigned int* arrayCPU, const unsigned int* arrayGPU, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        if(arrayCPU[i] != arrayGPU[i]) {
            printf("Arrays are not equal (arrayCPU[%u] = %d, arrayGPU[%u] = %d)\n", i, arrayCPU[i], i, arrayGPU[i]);
            return;
        }
    }
}

void radixSortCPU(const unsigned int* input, unsigned int* output, unsigned int N) {
    unsigned int* tempOutput = (unsigned int*) malloc(N*sizeof(unsigned int));

    //Partition around the LSB
    unsigned int firstOneIdx = 0;
    for(unsigned int i = 0; i < N; i++){
        if (!(input[i] & 1)) { 
            tempOutput[i] = tempOutput[firstOneIdx];
            tempOutput[firstOneIdx] = input[i];
            firstOneIdx++;
        }
        else{ tempOutput[i] = input[i]; }
    }

    std::swap(output, tempOutput);
    for(unsigned int b = 1; b < 32; b++){
        unsigned int numOnes = 0;
        for(unsigned int i = 0; i < N; i++){
            if ((output[i] >> b) & 1){ numOnes++; }
        }

        unsigned int curOnes = 0;
        for(unsigned int i = 0; i < N; i++){
            if ((output[i] >> b) & 1) {
                tempOutput[N-numOnes+curOnes] = output[i];
                curOnes++;
            }
            else{
                tempOutput[i - curOnes] = output[i];
            }
        }

        std::swap(output, tempOutput);
    }

    free(tempOutput);
}

void mergeSortCPU(const unsigned int* input, unsigned int* output, unsigned int N) {
    if (N == 0){ return; }
    if (N == 1){ output[0] = input[0]; return; }

    unsigned int mid = N / 2;
    mergeSortCPU(input, output, mid);
    mergeSortCPU(&input[mid], &output[mid], N - mid);

    unsigned int* A = (unsigned int*) malloc(mid*sizeof(unsigned int));
    unsigned int* B = (unsigned int*) malloc((N-mid)*sizeof(unsigned int));

    memcpy(A, output, mid*sizeof(unsigned int));
    memcpy(B, &output[mid], (N-mid)*sizeof(unsigned int));
    mergeCPU<unsigned int>(A, B, output, mid, N-mid);

    free(A); free(B);
}

// Sorts an array A using either radix sort (type 1) or merge sort (type 2)
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
	unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int N = (argc > 2) ? (atoi(argv[2])) : 10000000;
	
	if (type == 1){ printf("Running parallelized radix sort\n"); }
	else { printf("Running parallelized merge sort\n"); }
	
    unsigned int* input = (unsigned int*) malloc(N*sizeof(unsigned int));
    unsigned int* outputCPU = (unsigned int*) malloc(N*sizeof(unsigned int));
    unsigned int* outputGPU = (unsigned int*) malloc(N*sizeof(unsigned int));
    for (unsigned int i = 0; i < N; ++i){ input[i] = rand(); }
    input[0] = (1U << 31);
    
    // Compute on CPU
    startTime(&timer);
    if (type == 1) { radixSortCPU(input, outputCPU, N); }
    else { mergeSortCPU(input, outputCPU, N); }
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

    // Compute on GPU
    startTime(&timer);
    if (type == 1) { radixSortGPU(input, outputGPU, N); }
    else { mergeSortGPU<unsigned int>(input, outputGPU, N); }
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);

    // Verify result
    if (!checkIfSorted(outputCPU, N)){ printf("CPU array is not sorted\n"); }
	checkIfEqual(outputCPU, outputGPU, N);

    // Free memory
    free(input); 
    free(outputCPU); free(outputGPU);

    return 0;
}
