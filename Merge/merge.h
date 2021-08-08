#ifndef _MERGE_H_
#define _MERGE_H_

//Note that A and B are assumed to NOT overlap with C

template <typename T>
void mergeCPU(const T *A, const T *B, T *C, unsigned int n, unsigned int m);

template <typename T>
void mergeGPU(const T *A, const T *B, T *C, unsigned int n, unsigned int m, unsigned int type);

#endif