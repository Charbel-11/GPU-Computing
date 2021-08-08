#include "SpMV_CSR.h"
#include "SpMV_COO.h"
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

void SpMV_COO_CPU(const COOMatrix<float>& cooMatrix, const float* inVector, float* outVector){
    for(int i = 0; i < cooMatrix.numRows; i++){ outVector[i] = 0; }

    for(int i = 0; i < cooMatrix.numNonzeros; i++){
        unsigned int row = cooMatrix.rowIdxs[i], col = cooMatrix.colIdxs[i];
        outVector[row] += inVector[col] * cooMatrix.values[i];
    }
}

void SpMV_CSR_CPU(const CSRMatrix<float>& csrMatrix, const float* inVector, float* outVector){
    for(int i = 0; i < csrMatrix.numRows; i++){ outVector[i] = 0; }

    for(int row = 0; row < csrMatrix.numRows; row++){
        float sum = 0;
        for(unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; i++){
            unsigned int col = csrMatrix.colIdxs[i];
            sum += inVector[col] * csrMatrix.values[i];
        }
        outVector[row] = sum;
    }
}

// Multiplies a sparse matrix with a vector
// type 1: uses COO, type 2: uses CSR
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
	unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int numNonzeros = (argc > 2) ? (atoi(argv[2])) : 1000000;
    unsigned int numRows = (argc > 3) ? (atoi(argv[3])) : 100000;
    unsigned int numCols = (argc > 4) ? (atoi(argv[4])) : 100000;
	
    if (type == 1){ printf("Running sparse matrix-vector multiplication using the COO format\n"); }
	else { printf("Running sparse matrix-vector multiplication using the CSR format\n"); }
    
    float* inVector = (float*) malloc(numCols*sizeof(float));
    float* outVectorCPU = (float*) malloc(numRows*sizeof(float));
    float* outVectorGPU = (float*) malloc(numRows*sizeof(float));
	
    for(int i = 0; i < numCols; i++){ inVector[i] = 1.0f*rand()/RAND_MAX; }

    COOMatrix<float> cooMatrix(numRows, numCols, numNonzeros, false); 
    CSRMatrix<float> csrMatrix(numRows, numCols, numNonzeros, false);

    if (type == 1) { cooMatrix.generateRandomMatrix(); }
    else { csrMatrix.generateRandomMatrix(); }

    // Compute on CPU
    startTime(&timer);
    if (type == 1){ SpMV_COO_CPU(cooMatrix, inVector, outVectorCPU); }
    else{ SpMV_CSR_CPU(csrMatrix, inVector, outVectorCPU); }
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

	// Compute on GPU
    startTime(&timer);
    if (type == 1){ SpMV_COO_GPU<float>(cooMatrix, inVector, outVectorGPU); }
    else{ SpMV_CSR_GPU<float>(csrMatrix, inVector, outVectorGPU); }
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);

    // Verify result
	checkIfEqual(outVectorCPU, outVectorGPU, numRows);

    // Free memory
    free(inVector);
    free(outVectorCPU); free(outVectorGPU);

    return 0;
}

