#include "Lambert.cuh"

Vector cameraPos;
Vector lightPos;
Color lightColor;
float lightPower;
Color ambientLight;

__device__ 
Lambert::Lambert(const Color& diffuseColor)
	: Shader(diffuseColor)
{
}

__device__ 
Color Lambert::shade(Ray ray, const IntersectionData& data)
{
	Vector N = faceforward(ray.dir, data.normal);

	Color diffuseColor = _color;
	Color lightContrib = ambientLight;

	if (testVisibility(data.p + N * 1e-3, lightPos))
	{
		Vector lightDir = lightPos - data.p;
		lightDir.normalize();

		double cosTheta = dot(lightDir, N);
		lightContrib += lightColor * lightPower / (data.p - lightPos).lengthSqr() * cosTheta;
	}

	return diffuseColor * lightContrib;
}