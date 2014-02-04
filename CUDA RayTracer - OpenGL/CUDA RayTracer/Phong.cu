#include "Phong.cuh"

__device__
Phong::Phong(const Color& diffuseColor, double exponent, float strength)
     : Shader(diffuseColor)
	 , exponent(exponent)
	 , strength(strength)
{
}

__device__
Color Phong::shade(Ray ray, const IntersectionData& data)
{
	// turn the normal vector towards us (if needed):
	Vector N = faceforward(ray.dir, data.normal);

	Color diffuseColor = this->_color;
	
	Color lightContrib = scene->ambientLight;
	Color specular(0, 0, 0);
	
	for (int i = 0; i < scene->dev_lights.size(); ++i)
	{
		int numSamples = scene->dev_lights[i]->getNumSamples();
		Color avgColor(0, 0, 0);
		Color avgSpecular(0, 0, 0);

		for (int j = 0; j < numSamples; ++j)
		{
			Vector lightPos;
			Color lightColor;
			scene->dev_lights[i]->getNthSample(j, data.p, lightPos, lightColor);

			if (testVisibility(data.p + N * 1e-3, lightPos) && lightColor.intensity() != 0)
			{
				Vector lightDir = lightPos - data.p;
				lightDir.normalize();
		
				// get the Lambertian cosine of the angle between the geometry's normal and
				// the direction to the light. This will scale the lighting:
				double cosTheta = dot(lightDir, N);

				// baseLight is the light that "arrives" to the intersection point
				Color baseLight = lightColor  / (data.p - lightPos).lengthSqr();
				if (cosTheta > 0)
				{
					avgColor += baseLight * cosTheta;
				}
				//lightContrib += baseLight * cosTheta; // lambertian contribution
		
				// R = vector after the ray from the light towards the intersection point
				// is reflected at the intersection:
				Vector R = reflect(-lightDir, N);
		
				double cosGamma = dot(R, -ray.dir);
				if (cosGamma > 0)
					specular += baseLight * pow(cosGamma, exponent) * strength; // specular contribution
			}
		}
		lightContrib += avgColor / numSamples;
		specular += avgSpecular / numSamples;
	}

	// specular is not multiplied by diffuseColor, since we want the specular hilights to be
	// independent on the material color. I.e., a blue ball has white hilights
	// (this is true for most materials, and false for some, e.g. gold)
	return diffuseColor * lightContrib + specular;
}