#ifndef _MERGE_H_
#define _MERGE_H_

void mergeCPU(int* A, int *B, int *C, unsigned int n, unsigned int m);
void mergeGPU(int* A, int *B, int *C, unsigned int n, unsigned int m, unsigned int type);

#endif