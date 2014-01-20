#include "Fresnel.cuh"

// Schlick's approximation
__device__
static float fresnel(const Vector& i, const Vector& n, float ior)
{
	float f = sqr((1.0f - ior) / (1.0f + ior));
	float NdotI = (float) -dot(n, i);
	return f + (1.0f - f) * pow(1.0f - NdotI, 5.0f);
}

__device__
Fresnel::Fresnel(float ior)
	: ior(ior)
{
}

__device__
Color Fresnel::getTexColor(const Ray& ray, double u, double v, Vector& normal)
{
	// fresnel() expects the IOR_WE_ARE_ENTERING : IOR_WE_ARE_EXITING, so
	// in the case we're exiting the geometry, be sure to take the reciprocal
	float eta = ior;
	if (dot(normal, ray.dir) > 0)
	{
		eta = 1.0f / eta;
	}
	Vector N = faceforward(ray.dir, normal);
	float fr = fresnel(ray.dir, N, eta);
	return Color(fr, fr, fr);
}