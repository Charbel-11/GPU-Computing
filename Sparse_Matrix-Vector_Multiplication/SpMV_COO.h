#ifndef _SPMV_COO_H_
#define _SPMV_COO_H_

#include <set>
#include <random>

template <typename T>
struct COOMatrix{
    unsigned int numRows, numCols, numNonzeros;
    unsigned int *rowIdxs, *colIdxs; T* values; 
    bool inGPU, allocatedMemory = false;

    COOMatrix(unsigned int _numRows, unsigned int _numCols, unsigned int _numNonZeros, bool _inGPU):
    numRows(_numRows), numCols(_numCols), numNonzeros(_numNonZeros), inGPU(_inGPU) {}

    void allocateArrayMemory(){
         if (inGPU){
            cudaMalloc((void**) &rowIdxs, numNonzeros*sizeof(unsigned int)); 
            cudaMalloc((void**) &colIdxs, numNonzeros*sizeof(unsigned int));
            cudaMalloc((void**) &values, numNonzeros*sizeof(T));
        }
        else{
            rowIdxs = (unsigned int*) malloc(numNonzeros*sizeof(unsigned int));
            colIdxs = (unsigned int*) malloc(numNonzeros*sizeof(unsigned int));
            values = (T*) malloc(numNonzeros*sizeof(T));
        }
        allocatedMemory = true;
    }

    void generateRandomMatrix(){
        allocateArrayMemory();
        
        std::set<std::pair<unsigned int, unsigned int>> seen;
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> distribRows(0, numRows - 1), distribCol(0, numCols - 1);

        for(unsigned int i = 0; i < numNonzeros; i++){
            unsigned int curRow = 0, curCol = 0;
            do{
                curRow = distribRows(gen); 
                curCol = distribCol(gen);
            } while(seen.count({curRow, curCol}));
            seen.insert({curRow, curCol});

            rowIdxs[i] = curRow; colIdxs[i] = curCol;
            values[i] = 1.0f*rand()/RAND_MAX;
        }
    }

    ~COOMatrix(){
        if (!allocatedMemory){ return; }
        if (inGPU){
            cudaFree(rowIdxs); cudaFree(colIdxs); 
            cudaFree(values);
        }
        else{
            free(rowIdxs); free(colIdxs);
            free(values);
        }
    }
};

template <typename T>
void SpMV_COO_GPU(const COOMatrix<T>& cooMatrix, const T* inVector, T* outVector);

#endif