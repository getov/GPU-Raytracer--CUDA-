#ifndef TEXTURE_H
#define TEXTURE_H

#include "Color.cuh"
#include "Vector3D.cuh"

class Texture 
{
public:
	__device__
	virtual ~Texture() {}

	__device__
	virtual Color getTexColor(const Ray& ray, double u, double v, Vector& normal) = 0;
};

#endif