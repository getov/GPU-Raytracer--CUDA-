#ifndef FRESNEL_H
#define FRESNEL_H

#include "Texture.cuh"

class Fresnel : public Texture
{
private:
	float ior;

public:
	__device__
	Fresnel(float ior);

	__device__
	Color getTexColor(const Ray& ray, double u, double v, Vector& normal);
};

// Schlick's approximation
__device__
static float fresnel(const Vector& i, const Vector& n, float ior);

#endif