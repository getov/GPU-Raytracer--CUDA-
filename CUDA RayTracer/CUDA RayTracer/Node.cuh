#ifndef NODE_H
#define NODE_H

#include "IGeometry.cuh"
#include "IShader.cuh"
#include "Transform.cuh"

class Node 
{
public:
	Geometry* geom;
	Shader* shader;
	Transform transform;
	
	__device__ Node() {}

	__device__ 
	Node(Geometry* g, Shader* s)
	{ 
		geom = g; shader = s; 
	}

	__device__
	bool intersect(Ray ray, IntersectionData& data)
	{
		ray.start = transform.undoPoint(ray.start);
		ray.dir = transform.undoDirection(ray.dir);
		double oldDist = data.dist;
		double rayDirLength = ray.dir.length();
		data.dist *= rayDirLength;
		ray.dir.normalize();

		bool res = geom->intersect(ray, data);

		if (!res)
		{
			data.dist = oldDist;
			return false;
		}

		data.normal = normalize(transform.direction(data.normal));
		data.p = transform.point(data.p);
		data.dist /= rayDirLength;
		return true;
	}
};

#endif