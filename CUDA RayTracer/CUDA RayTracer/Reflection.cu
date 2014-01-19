#include "Reflection.cuh"
#include "cuda_renderer.cuh"
#include "Util.cuh"


__device__
void Reflection::getRandomDiscPoint(double& x, double& y)
{
	// pick a random point in the unit disc with uniform probability by using polar coords.
	// Note the sqrt(). For explanation why it's needed, see 
	// http://mathworld.wolfram.com/DiskPointPicking.html
	double theta = randomFloat() * 2 * PI;
	double rho = sqrt(randomFloat());
	x = rho * cos(theta);
	y = rho * sin(theta);
}

__device__
Reflection::Reflection(const Color& filter, double glossiness, int numSamples)
	: Shader(filter)
	, glossiness(glossiness)
	, numSamples(numSamples)
{
}

__device__
Color Reflection::shade(Ray ray, const IntersectionData& data)
{
	Vector N = faceforward(ray.dir, data.normal);
	
	if (glossiness == 1.0)
	{
		Vector reflected = reflect(ray.dir, N);
		
		Ray newRay = ray;
		newRay.start = data.p + N * 1e-3;
		newRay.dir = reflected;
		newRay.depth = ray.depth + 1;
		return raytrace(newRay) * _color;
	} 
	else // Not working at the moment. Maybe it need more stack size per thread than my GPU supports
	{
		Vector a, b;
		orthonormedSystem(N, a, b);
		Color result(0, 0, 0);
		double scaling = tan((1 - glossiness) * PI/2);
		int samplesWanted = ray.depth == 0 ? numSamples : 5;
		for (int i = 0; i < samplesWanted; i++)
		{
			Vector reflected;
			do 
			{
				double x, y;
				getRandomDiscPoint(x, y);
				x *= scaling;
				y *= scaling;
				
				Vector newNormal = N + a * x + b * y;
				newNormal.normalize();
				
				reflected = reflect(ray.dir, newNormal);
			} while (dot(reflected, N) < 0);
			
			Ray newRay = ray;
			newRay.start = data.p + N * 1e-6;
			newRay.dir = reflected;
			newRay.depth = ray.depth + 1;
			result += raytrace(newRay) * _color;
		}
		return result ;
	}
}