#ifndef NODE_H
#define NODE_H

#include "IGeometry.cuh"
#include "IShader.cuh"

class Node 
{
public:
	Geometry* geom;
	Shader* shader;
	
	__device__ Node() {}

	__device__ 
	Node(Geometry* g, Shader* s)
	{ 
		geom = g; shader = s; 
	}
};

#endif