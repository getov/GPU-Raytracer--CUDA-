#ifndef REFRACTION_H
#define REFRACTION_H

#include "IShader.cuh"
#include "Node.cuh"

class Refraction : public Shader
{
private:
	float ior;

public:
	__device__
	Refraction(const Color& filter, float ior);

	__device__
	Color shade(Ray ray, const IntersectionData& data);
				
};

#endif