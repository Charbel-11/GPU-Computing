#ifndef _COMMON_H_
#define _COMMON_H_

#define NUM_BINS 256

void histogramGPU(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height, unsigned int type);

#endif