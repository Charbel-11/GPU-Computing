#ifndef _SPMV_ELL_H_
#define _SPMV_ELL_H_

#include <set>
#include <random>
#include <unordered_map>

template <typename T>
struct ELLMatrix{
    unsigned int numRows, numCols, numNonzeros, maxNonzerosPerRow = 0;
    unsigned int *colIdxs, *nonZeroPerRow; T* values;
    bool inGPU, allocatedMemory = false;

    ELLMatrix(unsigned int _numRows, unsigned int _numCols, unsigned int _numNonzeros, bool _inGPU):
    numRows(_numRows), numCols(_numCols), numNonzeros(_numNonzeros), inGPU(_inGPU) {}

    void allocateArrayMemory(){
        if (inGPU){
            cudaMalloc((void**) &nonZeroPerRow, numRows*sizeof(unsigned int));
            cudaMalloc((void**) &colIdxs, maxNonzerosPerRow*numRows*sizeof(unsigned int));
            cudaMalloc((void**) &values, maxNonzerosPerRow*numRows*sizeof(T));
        }
        else{
            nonZeroPerRow = (unsigned int*) malloc(numRows*sizeof(unsigned int));
            colIdxs = (unsigned int*) malloc(maxNonzerosPerRow*numRows*sizeof(unsigned int));
            values = (T*) malloc(maxNonzerosPerRow*numRows*sizeof(T));
        }
        allocatedMemory = true;
    }

    void generateRandomMatrix(){
        std::set<std::pair<unsigned int, unsigned int>> seen;
        std::unordered_map<unsigned int, unsigned int> rowFreq;
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> distribRows(0, numRows - 1), distribCol(0, numCols - 1);

        for(unsigned int i = 0; i < numNonzeros; i++){
            unsigned int curRow = 0, curCol = 0;
            do{
                curRow = distribRows(gen); 
                curCol = distribCol(gen);
            } while(seen.count({curRow, curCol}));
            seen.insert({curRow, curCol});

            rowFreq[curRow]++;
            maxNonzerosPerRow = (maxNonzerosPerRow > rowFreq[curRow] ? maxNonzerosPerRow : rowFreq[curRow]);
        }

        allocateArrayMemory();

        for(unsigned int i = 0; i < maxNonzerosPerRow*numRows; i++){
            colIdxs[i] = numCols + 1;
            values[i] = 0;
        }
        for(unsigned int i = 0; i < numRows; i++){ nonZeroPerRow[i] = 0; }

        unsigned int curCnt = 0, lastRow = 0;
        for(auto &p : seen){
            unsigned int curRow = p.first;
            if (lastRow != curRow) { 
                nonZeroPerRow[lastRow] = curCnt;
                lastRow = curRow; curCnt = 0; 
            }
            curCnt++;

            unsigned int idx = curRow + numRows * (curCnt - 1);
            colIdxs[idx] = p.second;
            values[idx] = 1.0f*rand()/RAND_MAX;
        }
        nonZeroPerRow[lastRow] = curCnt;
    }

    ~ELLMatrix(){
        if (!allocatedMemory){ return; }
        if (inGPU){
            cudaFree(colIdxs); cudaFree(nonZeroPerRow);
            cudaFree(values);
        }
        else{
            free(colIdxs); free(nonZeroPerRow);
            free(values);
        }
    }
};

template <typename T>
void SpMV_ELL_GPU(const ELLMatrix<T>& ellMatrix, const T* inVector, T* outVector);

#endif