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
	
	Color lightContrib = ambientLight;
	Color specular(0, 0, 0);
	
	if (testVisibility(data.p + N * 1e-3, lightPos))
	{
		Vector lightDir = lightPos - data.p;
		lightDir.normalize();
		
		// get the Lambertian cosine of the angle between the geometry's normal and
		// the direction to the light. This will scale the lighting:
		double cosTheta = dot(lightDir, N);

		// baseLight is the light that "arrives" to the intersection point
		Color baseLight = lightColor * lightPower / (data.p - lightPos).lengthSqr();
		
		lightContrib += baseLight * cosTheta; // lambertian contribution
		
		// R = vector after the ray from the light towards the intersection point
		// is reflected at the intersection:
		Vector R = reflect(-lightDir, N);
		
		double cosGamma = dot(R, -ray.dir);
		if (cosGamma > 0)
			specular += baseLight * pow(cosGamma, exponent) * strength; // specular contribution
	}
	// specular is not multiplied by diffuseColor, since we want the specular hilights to be
	// independent on the material color. I.e., a blue ball has white hilights
	// (this is true for most materials, and false for some, e.g. gold)
	return diffuseColor * lightContrib + specular;
}