#include "Lambert.cuh"

//Vector cameraPos;
//Vector lightPos;
//Color lightColor;
//float lightPower;
//Color ambientLight;

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
	Color lightContrib = scene->ambientLight;

	for (int i = 0; i < scene->dev_lights.size(); ++i)
	{
		int numSamples = scene->dev_lights[i]->getNumSamples();
		Color avgColor(0, 0, 0);

		for (int j = 0; j < numSamples; ++j)
		{
			Vector lightPos;
			Color lightColor;
			scene->dev_lights[i]->getNthSample(j, data.p, lightPos, lightColor);

			if (testVisibility(data.p + N * 1e-3, lightPos) && lightColor.intensity() != 0)
			{
				Vector lightDir = lightPos - data.p;
				lightDir.normalize();

				double cosTheta = dot(lightDir, N);
				if (cosTheta > 0)
				{
					avgColor += lightColor / (data.p - lightPos).lengthSqr() * cosTheta;
				}
				//lightContrib += lightColor * lightPower / (data.p - lightPos).lengthSqr() * cosTheta;
			}
		}
		lightContrib += avgColor / numSamples;
	}
	
	return diffuseColor * lightContrib;
}