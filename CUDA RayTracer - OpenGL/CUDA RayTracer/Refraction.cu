#include "Refraction.cuh"
#include "cuda_renderer.cuh"

__device__
Refraction::Refraction(const Color& filter, float ior)
	: Shader(filter)
	, ior(ior)
{
}

__device__
Color Refraction::shade(Ray ray, const IntersectionData& data)
{
	Vector N = faceforward(ray.dir, data.normal);
	
	float eta = ior;
	if (dot(ray.dir, data.normal) < 0)
	{
		eta = 1.0f / eta;
	}
	
	Vector refracted = refract(ray.dir, N, eta);
	
	// total inner refraction:
	if (refracted.lengthSqr() == 0)
	{
		return Color(0, 0, 0);
	}
	
	Ray newRay = ray;
	newRay.start = data.p + ray.dir * 1e-3;
	newRay.dir = refracted;
	newRay.depth = ray.depth + 1;
	
	return raytrace(newRay) * _color;
}