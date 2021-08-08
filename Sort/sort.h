#ifndef _SORT_H_
#define _SORT_H_

void radixSortGPU(const unsigned int* input, unsigned int* output, unsigned int N);

template <typename T>
void mergeSortGPU(const T* input, T* output, unsigned int N);

#endif