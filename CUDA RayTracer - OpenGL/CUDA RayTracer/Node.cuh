#ifndef NODE_H
#define NODE_H

#include "IGeometry.cuh"
#include "IShader.cuh"
#include "Transform.cuh"
#include "Texture.cuh"
class Shader;

class Node 
{
public:
	Geometry* geom;
	Shader* shader;
	Transform transform;
	Texture* bumpTex;
	GeometryID nodeID;
	
	__device__ Node() {}

	__device__ 
	Node(Geometry* g, Shader* s, Texture* bump = nullptr)
	{ 
		geom    = g;
		shader  = s; 
		bumpTex = bump;
	}

	__device__
	bool intersect(Ray ray, IntersectionData& data)
	{
		// transform from world space into model space
		Ray modelRay;
		modelRay.start = transform.undoPoint(ray.start);
		modelRay.dir = transform.undoDirection(ray.dir);
		modelRay.depth = ray.depth;

		double oldDist = data.dist;
		double rayDirLength = ray.dir.length();
		data.dist *= rayDirLength;
		modelRay.dir.normalize();

		if (!geom->intersect(modelRay, data))
		{
			data.dist = oldDist;
			return false;
		}

		// transform from model space into world space
		data.normal = normalize(transform.direction(data.normal));
		data.dNdx = normalize(transform.direction(data.dNdx));
		data.dNdy = normalize(transform.direction(data.dNdy));
		data.p = transform.point(data.p);
		data.dist /= rayDirLength;
		nodeID = data.geomID;

		return true;
	}
};

#endif