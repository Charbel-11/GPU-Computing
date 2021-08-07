#include "histogram.h"
#include "../Helper_Code/timer.h"

void checkIfEqual(unsigned int* binsCPU, unsigned int* binsGPU) {
    for (unsigned int b = 0; b < NUM_BINS; b++) {
        if(binsCPU[b] != binsGPU[b]) {
            printf("Histograms are not equal (binCPU[%u] = %u, binGPU[%u] = %u)\n", b, binsCPU[b], b, binsGPU[b]);
            return;
        }
    }
}

void histogramCPU(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    for(unsigned int i = 0; i < width*height; i++) {
        unsigned char b = image[i];
        bins[b]++;
    }
}

// A histogram divides the data into bins and finds the occurrence of values in each bin
// In this case, the data is an image (a collection of pixels) so the histogram is a color histogram
// type 1 runs a basic parallelized algorithm, type 2 uses privatization and shared memory, type 3 also uses thread coarsening
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int height = (argc > 2) ? (atoi(argv[2])) : 8000;
    unsigned int width = (argc > 3) ? (atoi(argv[3])) : 8000;

    if (type == 1){ printf("Running parallelized histogram\n"); }
	else if (type == 2) { printf("Running parallelized histogram with privatization and shared memory\n"); }
    else { printf("Running parallelized histogram with privatization and shared memory and thread coarsening\n"); }

    unsigned char* image = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    unsigned int* binsCPU = (unsigned int*) malloc(NUM_BINS*sizeof(unsigned int));
    unsigned int* binsGPU = (unsigned int*) malloc(NUM_BINS*sizeof(unsigned int));
    for (unsigned int row = 0; row < height; row++)
        for (unsigned int col = 0; col < width; col++)
            image[row*width + col] = rand()%256;
    memset(binsCPU, 0, NUM_BINS*sizeof(unsigned int));

    // Compute on CPU
    startTime(&timer);
    histogramCPU(image, binsCPU, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

    // Compute on GPU
    startTime(&timer);
    histogramGPU(image, binsGPU, width, height, type);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);    

    // Verify result
    checkIfEqual(binsCPU, binsGPU);

    // Free memory
    free(image);
    free(binsCPU); free(binsGPU);

    return 0;
}
