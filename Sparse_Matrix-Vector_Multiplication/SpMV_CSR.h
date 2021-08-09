#ifndef _SPMV_CSR_H_
#define _SPMV_CSR_H_

#include <set>
#include <random>

template <typename T>
struct CSRMatrix{
    unsigned int numRows, numCols, numNonzeros;
    unsigned int *rowPtrs, *colIdxs; T* values;
    bool inGPU, allocatedMemory = false;

    CSRMatrix(unsigned int _numRows, unsigned int _numCols, unsigned int _numNonZeros, bool _inGPU):
    numRows(_numRows), numCols(_numCols), numNonzeros(_numNonZeros), inGPU(_inGPU) { }

    void allocateArrayMemory(){
        if (inGPU){
            cudaMalloc((void**) &rowPtrs, (numRows + 1)*sizeof(unsigned int)); 
            cudaMalloc((void**) &colIdxs, numNonzeros*sizeof(unsigned int));
            cudaMalloc((void**) &values, numNonzeros*sizeof(T));
        }
        else{
            rowPtrs = (unsigned int*) malloc((numRows + 1)*sizeof(unsigned int));
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
        }

        unsigned int curRow = 1, cnt = 0; rowPtrs[0] = 0;
        for(auto &p : seen){
            while (p.first >= curRow){ rowPtrs[curRow++] = cnt; }
            colIdxs[cnt] = p.second;
            values[cnt] = 1.0f*rand()/RAND_MAX;
            cnt++; 
        }
        while(curRow <= numRows){ rowPtrs[curRow++] = cnt; }
    }

    ~CSRMatrix(){
        if (!allocatedMemory){ return; }
        if (inGPU){
            cudaFree(rowPtrs); cudaFree(colIdxs); 
            cudaFree(values);
        }
        else{
            free(rowPtrs); free(colIdxs);
            free(values);
        }
    }
};

template <typename T>
void SpMV_CSR_GPU(const CSRMatrix<T>& csrMatrix, const T* inVector, T* outVector);

#endif