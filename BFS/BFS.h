#ifndef _BFS_H_
#define _BFS_H_

#include "GraphCSR.h"

void BFS_VertexCentric_TopDown(const GraphCSR& graphCSR, unsigned int* dist, unsigned int src);

#endif