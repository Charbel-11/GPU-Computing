//This is a representation of the adjacency matrix of unweighted and undirected graphs using the CSR format

#ifndef _GRAPH_CSR_H_
#define _GRAPH_CSR_H_

#include <set>
#include <random>

struct GraphCSR{
    unsigned int numNodes, numEdges;
    unsigned int *srcPtrs, *dest;
    bool inGPU;

    // numEdges is even since each edge is added in both directions and we have no self-loop
    GraphCSR(unsigned int _n, unsigned int _m, bool _inGPU):
    numNodes(_n), numEdges(_m), inGPU(_inGPU) {
        if (numEdges & 1) { numEdges++; }
    }

    void allocateMemory(){
        if (inGPU){
            cudaMalloc((void**) &srcPtrs, (numNodes + 1)*sizeof(unsigned int)); 
            cudaMalloc((void**) &dest, numEdges*sizeof(unsigned int));
        }
        else{
            srcPtrs = (unsigned int*) malloc((numNodes + 1)*sizeof(unsigned int));
            dest = (unsigned int*) malloc(numEdges*sizeof(unsigned int));
        }
    }

    void deallocateMemory(){
        if (inGPU){
            cudaFree(srcPtrs); cudaFree(dest); 
        }
        else{
            free(srcPtrs); free(dest);
        }
    }

    void generateRandomSymmetricMatrix(){        
        std::set<std::pair<unsigned int, unsigned int>> seen;
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> distribRows(0, numNodes - 1), distribCol(0, numNodes - 1);

        for(unsigned int i = 0; i < numEdges/2; i++){
            unsigned int curRow = 0, curCol = 0;
            do{
                curRow = distribRows(gen); 
                curCol = distribCol(gen);
            } while(seen.count({curRow, curCol}) || curRow == curCol);
            seen.insert({curRow, curCol});
            seen.insert({curCol, curRow});
        }

        convertPairsToCSR(seen);
    }

    private:
    void convertPairsToCSR(const std::set<std::pair<unsigned int, unsigned int>>& pairs){
        allocateMemory();

        unsigned int curRow = 1, cnt = 0; srcPtrs[0] = 0;
        for(auto &p : pairs){
            while (p.first >= curRow){ srcPtrs[curRow++] = cnt; }
            dest[cnt++] = p.second;
        }
        while(curRow <= numNodes){ srcPtrs[curRow++] = cnt; }
    }
};

#endif