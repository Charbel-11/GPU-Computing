#ifndef _MERGE_H_
#define _MERGE_H_

void mergeCPU(const int* A, const int *B, int *C, unsigned int n, unsigned int m);
void mergeGPU(const int* A, const int *B, int *C, unsigned int n, unsigned int m, unsigned int type);

#endif