#include "BFS.h"
#include "../Helper_Code/timer.h"
#include <math.h>

#define BLOCK_DIM 128

__global__ void BFS_VertexCentric_TopDown_Kernel(const GraphCSR graphCSR, unsigned int* dist, unsigned int* newVertexWasVisited, unsigned int curLevel){
    unsigned int curNode = blockIdx.x * blockDim.x + threadIdx.x;
    if (curNode >= graphCSR.numNodes || dist[curNode] != curLevel) { return; }

    if (curNode == 0){
        for(int i = 0; i < 100; i++){
            printf("%u ", graphCSR.dest[graphCSR.numEdges - i - 1]);
        } 
        printf("\n");
    }

    printf("Calling at %u %u\n", curNode, curLevel);
    printf("%u ", graphCSR.dest[64168]);
    printf("%u: %u %u %u\n", curNode, graphCSR.srcPtrs[curNode], graphCSR.srcPtrs[curNode+1], graphCSR.dest[graphCSR.srcPtrs[curNode]]);

    for(unsigned int edge = graphCSR.srcPtrs[curNode]; edge < graphCSR.srcPtrs[curNode + 1]; edge++){
        unsigned int curNeighbor = graphCSR.dest[edge];
        printf("%u %u\n", curNode, curNeighbor);
        if (dist[curNeighbor] == UINT_MAX){
            dist[curNeighbor] = curLevel + 1;
            *newVertexWasVisited = 1;
        }
    }
}

__global__ void BFS_VertexCentric_BottomUp_Kernel(const GraphCSR graphCSR, unsigned int* dist, unsigned int* newVertexWasVisited, unsigned int curLevel){
    unsigned int curNode = blockIdx.x * blockDim.x + threadIdx.x;
    if (curNode >= graphCSR.numNodes || dist[curNode] != UINT_MAX) { return; }

    printf("Calling at %u %u\n", curNode, curLevel);

    for(unsigned int edge = graphCSR.srcPtrs[curNode]; edge < graphCSR.srcPtrs[curNode + 1]; edge++){
        unsigned int curNeighbor = graphCSR.dest[edge];
        if (dist[curNeighbor] == curLevel){
            dist[curNode] = curLevel + 1;
            *newVertexWasVisited = 1;
            break;
        }
    }
}

void BFS_VertexCentric_Helper(const GraphCSR& graphCSR_d, unsigned int* dist_d, unsigned int* newVertexWasVisited_d, unsigned int src, unsigned int type){
    unsigned int numBlocks = (graphCSR_d.numNodes + BLOCK_DIM - 1) / BLOCK_DIM;

    unsigned int newVertexWasVisited = 1;
    for(unsigned int curLevel = 0; newVertexWasVisited == 1; curLevel++){
        newVertexWasVisited = 0;
        cudaMemcpy(newVertexWasVisited_d, &newVertexWasVisited, sizeof(unsigned int), cudaMemcpyHostToDevice); 

        if (type == 1) { BFS_VertexCentric_TopDown_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel); }
        else if (type == 2) { BFS_VertexCentric_BottomUp_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel);  }
        else{
            if (curLevel <= log(graphCSR_d.numNodes)){ BFS_VertexCentric_TopDown_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel); }
            else { BFS_VertexCentric_BottomUp_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel); }
        }

        cudaMemcpy(&newVertexWasVisited, newVertexWasVisited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
}

void BFS_VertexCentric(const GraphCSR& graphCSR, unsigned int* dist, unsigned int src, unsigned int type){
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    GraphCSR graphCSR_d(graphCSR.numNodes, graphCSR.numEdges, true);
    printf("%u %u\n", graphCSR_d.numNodes, graphCSR_d.numEdges);    

    graphCSR_d.allocateArrayMemory();
    unsigned int *dist_d, *newVertexWasVisited_d; 
    cudaMalloc((void**) &dist_d, graphCSR_d.numNodes*sizeof(unsigned int)); 
    cudaMalloc((void**) &newVertexWasVisited_d, sizeof(unsigned int)); 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

    for(int i = 0; i < 100; i++){
        printf("%u ", graphCSR.dest[graphCSR.numEdges - i - 1]);
    } printf("\n");

    //Copying data to GPU from Host
    startTime(&timer);
    cudaMemcpy(graphCSR_d.srcPtrs, graphCSR.srcPtrs, (graphCSR_d.numNodes + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice); 
    cudaMemcpy(graphCSR_d.dest, graphCSR.dest, graphCSR_d.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    for(int i = 0; i < graphCSR_d.numNodes; i++){ dist[i] = UINT_MAX; } dist[src] = 0;
    cudaMemcpy(dist_d, dist, graphCSR_d.numNodes*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying to GPU time");

    //Calling kernel
    startTime(&timer);
    BFS_VertexCentric_Helper(graphCSR_d, dist_d, newVertexWasVisited_d, src, type);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time", GREEN);
	
	//Copying data from GPU to Host
    startTime(&timer);
    cudaMemcpy(dist, dist_d, graphCSR_d.numNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copying from GPU time");

	//Freeing GPU memory
    startTime(&timer);
    cudaFree(dist_d); cudaFree(newVertexWasVisited_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Deallocation time");
}
