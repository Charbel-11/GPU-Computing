#ifndef _MERGE_H_
#define _MERGE_H_

template <typename T>
void mergeCPU(const T *A, const T *B, T *C, unsigned int n, unsigned int m);

template <typename T>
void mergeGPU(const T *A, const T *B, T *C, unsigned int n, unsigned int m, unsigned int type);

template <typename T>
void mergeGPUOnDevice(const T* A_d, const T *B_d, T *C_d, unsigned int n, unsigned int m, unsigned int type);

#endif