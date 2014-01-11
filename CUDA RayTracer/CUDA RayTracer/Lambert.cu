#include "Lambert.cuh"

Vector cameraPos;
Vector lightPos;
Color lightColor;
float lightPower;
Color ambientLight;

__device__ Lambert::Lambert(const Color& diffuseColor)
	: Shader(diffuseColor)
{
}

__device__ Color Lambert::shade(Ray ray, const IntersectionData& data)
{
	Color result = _color;
	
	result = result * lightColor * lightPower / (data.p - lightPos).lengthSqr();
	Vector lightDir = lightPos - data.p;
	lightDir.normalize();
	
	double cosTheta = dot(lightDir, data.normal);
	result = result * cosTheta;

	return result;
}