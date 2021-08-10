#include "../Helper_Code/timer.h"
#include "BFS.h"
#include <queue>
#include <vector>

void checkIfEqual(unsigned int* arrayCPU, unsigned int* arrayGPU, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        if(arrayCPU[i] != arrayGPU[i]) {
            printf("Arrays are not equal (arrayCPU[%u] = %u, arrayGPU[%u] = %u)\n", i, arrayCPU[i], i, arrayGPU[i]);
            return;
        }
    }
}

void BFS_CPU(const GraphCSR& graphCSR, unsigned int* dist, unsigned int src){
    std::queue<unsigned int> Q;
    std::vector<bool> visited(graphCSR.numNodes, false);
    Q.push(src); visited[src] = true; dist[src] = 0;

    while(!Q.empty()){
        unsigned int curNode = Q.front(); Q.pop();
        for(unsigned int edge = graphCSR.srcPtrs[curNode]; edge < graphCSR.srcPtrs[curNode + 1]; edge++){
            unsigned int curNeighbor = graphCSR.dest[edge];
            if (!visited[curNeighbor]){
                visited[curNeighbor] = true;
                dist[curNeighbor] = dist[curNode] + 1;
                Q.push(curNeighbor); 
            }
        }
    }
}

// BFS takes an unweighted graph and a source node and finds the min distance between that source node and every other node
// type 1 uses a top-down vertex-centric algorithm, type 2 uses
int main(int argc, char**argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int type = (argc > 1) ? (atoi(argv[1])) : 1;
    unsigned int numNodes = (argc > 2) ? (atoi(argv[2])) : 100000;
    unsigned int numEdges = (argc > 3) ? (atoi(argv[3])) : 500000;

    if (type == 1){ printf("Running a top-down vertex-centric parallelized BFS\n"); }

    unsigned int src = 0;
    unsigned int* distCPU = (unsigned int*) malloc(numNodes*sizeof(unsigned int));
    unsigned int* distGPU = (unsigned int*) malloc(numNodes*sizeof(unsigned int));
    for(int i = 0; i < numNodes; i++){ distCPU[i] = distGPU[i] = UINT_MAX; }

    GraphCSR graphCSR(numNodes, numEdges, false); 
    graphCSR.generateRandomSymmetricMatrix();

    // Compute on CPU
    startTime(&timer);
    BFS_CPU(graphCSR, distCPU, src);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", BLUE);

    // Compute on GPU
    startTime(&timer);
    if (type == 1) { BFS_VertexCentric_TopDown(graphCSR, distGPU, src); }
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", RED);    

    // Verify result
    checkIfEqual(distCPU, distGPU, numNodes);

    // Free memory
    free(distCPU); free(distGPU);

    return 0;
}
