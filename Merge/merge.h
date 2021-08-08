#ifndef _MERGE_H_
#define _MERGE_H_

template <typename T>
void mergeCPU(const T *A, const T *B, T *C, unsigned int n, unsigned int m);

template <typename T>
void mergeGPU(const T *A, const T *B, T *C, unsigned int n, unsigned int m, unsigned int type);

#endif