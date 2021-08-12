#include "BFS.h"
#include "../Helper_Code/timer.h"
#include <math.h>
#include <algorithm>

#define BLOCK_DIM 128
#define LOCAL_QUEUE_CAP 2048

__global__ void BFS_VertexCentric_TopDown_Kernel(const GraphCSR graphCSR, unsigned int* dist, unsigned int* newVertexWasVisited, unsigned int curLevel){
    unsigned int curNode = blockIdx.x * blockDim.x + threadIdx.x;
    if (curNode >= graphCSR.numNodes || dist[curNode] != curLevel) { return; }

    for(unsigned int edge = graphCSR.srcPtrs[curNode]; edge < graphCSR.srcPtrs[curNode + 1]; edge++){
        unsigned int curNeighbor = graphCSR.dest[edge];
        if (dist[curNeighbor] == UINT_MAX){
            dist[curNeighbor] = curLevel + 1;
            *newVertexWasVisited = 1;
        }
    }
}

__global__ void BFS_VertexCentric_BottomUp_Kernel(const GraphCSR graphCSR, unsigned int* dist, unsigned int* newVertexWasVisited, unsigned int curLevel){
    unsigned int curNode = blockIdx.x * blockDim.x + threadIdx.x;
    if (curNode >= graphCSR.numNodes || dist[curNode] != UINT_MAX) { return; }

    for(unsigned int edge = graphCSR.srcPtrs[curNode]; edge < graphCSR.srcPtrs[curNode + 1]; edge++){
        unsigned int curNeighbor = graphCSR.dest[edge];
        if (dist[curNeighbor] == curLevel){
            dist[curNode] = curLevel + 1;
            *newVertexWasVisited = 1;
            break;
        }
    }
}

__global__ void BFS_VertexCentric_FrontierBased_Kernel(const GraphCSR graphCSR, unsigned int* dist, unsigned int* curFrontier, unsigned int* nextFrontier, 
    unsigned int numCurFrontier, unsigned int* numNextFrontier, unsigned int curLevel){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCurFrontier) { return; }

    unsigned int curNode = curFrontier[i];
    for(unsigned int edge = graphCSR.srcPtrs[curNode]; edge < graphCSR.srcPtrs[curNode + 1]; edge++){
        unsigned int curNeighbor = graphCSR.dest[edge];
        if (atomicCAS(&dist[curNeighbor], UINT_MAX, curLevel + 1) == UINT_MAX){
            unsigned int nextFrontierIdx = atomicAdd(numNextFrontier, 1);
            nextFrontier[nextFrontierIdx] = curNeighbor;
        }
    }
}

__global__ void BFS_VertexCentric_FrontierBasedWithPrivatization_Kernel(const GraphCSR graphCSR, unsigned int* dist, unsigned int* curFrontier, 
    unsigned int* nextFrontier, unsigned int numCurFrontier, unsigned int* numNextFrontier, unsigned int curLevel){
    __shared__ unsigned int nextFrontier_s[LOCAL_QUEUE_CAP];
    __shared__ unsigned int numNextFrontier_s;
    if (threadIdx.x == 0){ numNextFrontier_s = 0; }
    __syncthreads();
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCurFrontier) { 
        unsigned int curNode = curFrontier[i];
        for(unsigned int edge = graphCSR.srcPtrs[curNode]; edge < graphCSR.srcPtrs[curNode + 1]; edge++){
            unsigned int curNeighbor = graphCSR.dest[edge];
            if (atomicCAS(&dist[curNeighbor], UINT_MAX, curLevel + 1) == UINT_MAX){
                unsigned int nextFrontierIdx_s = atomicAdd(&numNextFrontier_s, 1);
                if (nextFrontierIdx_s < LOCAL_QUEUE_CAP) { nextFrontier_s[nextFrontierIdx_s] = curNeighbor; }
                else{
                    numNextFrontier_s = LOCAL_QUEUE_CAP;
                    unsigned int nextFrontierIdx = atomicAdd(numNextFrontier, 1);
                    nextFrontier[nextFrontierIdx] = curNeighbor;
                }
            }
        }
    }
    __syncthreads();

    __shared__ unsigned int nextFrontierStartIdx;
    if (threadIdx.x == 0) { nextFrontierStartIdx = atomicAdd(numNextFrontier, numNextFrontier_s); }
    __syncthreads();

    for(unsigned int nextFrontierIdx_s = threadIdx.x; nextFrontierIdx_s < numNextFrontier_s; nextFrontierIdx_s += blockDim.x){
        unsigned int nextFrontierIdx = nextFrontierStartIdx + nextFrontierIdx_s;
        nextFrontier[nextFrontierIdx] = nextFrontier_s[nextFrontierIdx_s];
    }
}

void BFS_VertexCentric_Helper(const GraphCSR& graphCSR_d, unsigned int* dist_d, unsigned int* newVertexWasVisited_d, unsigned int src, unsigned int type){
    unsigned int numBlocks = (graphCSR_d.numNodes + BLOCK_DIM - 1) / BLOCK_DIM;

    unsigned int newVertexWasVisited = 1;
    for(unsigned int curLevel = 0; newVertexWasVisited; curLevel++){
        newVertexWasVisited = 0;
        cudaMemcpy(newVertexWasVisited_d, &newVertexWasVisited, sizeof(unsigned int), cudaMemcpyHostToDevice); 

        if (type == 1) { BFS_VertexCentric_TopDown_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel); }
        else if (type == 2) { BFS_VertexCentric_BottomUp_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel);  }
        else {
            if (curLevel <= log(graphCSR_d.numNodes)){ BFS_VertexCentric_TopDown_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel); }
            else { BFS_VertexCentric_BottomUp_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, newVertexWasVisited_d, curLevel); }
        }

        cudaMemcpy(&newVertexWasVisited, newVertexWasVisited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
}

void BFS_VertexCentric_FrontierBased_Helper(const GraphCSR& graphCSR_d, unsigned int* dist_d, unsigned int* newVertexWasVisited_d, unsigned int src, unsigned int type){
    unsigned int *buffer1_d, *buffer2_d;
    unsigned int *numNextFrontier_d; 
    cudaMalloc((void**) &buffer1_d, graphCSR_d.numNodes*sizeof(unsigned int)); 
    cudaMalloc((void**) &buffer2_d, graphCSR_d.numNodes*sizeof(unsigned int)); 
    cudaMalloc((void**) &numNextFrontier_d, sizeof(unsigned int)); 

    unsigned int *curFrontier_d = buffer1_d, *nextFrontier_d = buffer2_d;
    cudaMemcpy(curFrontier_d, &src, sizeof(unsigned int), cudaMemcpyHostToDevice); 

    unsigned int numCurFrontier = 1;
    for(unsigned int curLevel = 0; numCurFrontier; curLevel++){
        cudaMemset(numNextFrontier_d, 0, sizeof(unsigned int));
        
        unsigned int numBlocks = (numCurFrontier + BLOCK_DIM - 1) / BLOCK_DIM;
        if (type == 4) { BFS_VertexCentric_FrontierBased_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, curFrontier_d, nextFrontier_d, numCurFrontier, numNextFrontier_d, curLevel); }
        else { BFS_VertexCentric_FrontierBasedWithPrivatization_Kernel <<< numBlocks, BLOCK_DIM >>> (graphCSR_d, dist_d, curFrontier_d, nextFrontier_d, numCurFrontier, numNextFrontier_d, curLevel); }
    
        std::swap(curFrontier_d, nextFrontier_d);
        cudaMemcpy(&numCurFrontier, numNextFrontier_d, sizeof(unsigned int), cudaMemcpyDeviceToHost); 
    }

    cudaFree(buffer1_d); cudaFree(buffer2_d);
    cudaFree(numNextFrontier_d);
}

void BFS_VertexCentric(const GraphCSR& graphCSR, unsigned int* dist, unsigned int src, unsigned int type){
    Timer timer;

	// Allocating GPU memory
    startTime(&timer);
    GraphCSR graphCSR_d(graphCSR.numNodes, graphCSR.numEdges, true);
    graphCSR_d.allocateMemory();
    unsigned int *dist_d, *newVertexWasVisited_d; 
    cudaMalloc((void**) &dist_d, graphCSR_d.numNodes*sizeof(unsigned int)); 
    cudaMalloc((void**) &newVertexWasVisited_d, sizeof(unsigned int)); 
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Allocation time");

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
    if (type >= 1 && type <= 3) { BFS_VertexCentric_Helper(graphCSR_d, dist_d, newVertexWasVisited_d, src, type); }
    else { BFS_VertexCentric_FrontierBased_Helper(graphCSR_d, dist_d, newVertexWasVisited_d, src, type); }
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
    graphCSR_d.deallocateMemory();
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Deallocation time");
}
