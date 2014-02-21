#include "Light.cuh"
#include "Util.cuh"

// Point Light

__device__
PointLight::PointLight(const Vector& position, const Color& color, const float& power)
	: Light(color, power)
	, pos(position)
{
}

__device__
int PointLight::getNumSamples()
{
	return 1;
}

__device__
void PointLight::getNthSample(int sampleIdx, const Vector& shadePos, Vector& samplePos, Color& color)
{
	samplePos = pos;
	color = this->m_color * this->m_power;
}

__device__
bool PointLight::intersect(const Ray& ray, double& intersectionDist)
{
	return false; // you can't intersect a point light
}

__device__
float PointLight::solidAngle(const Vector& x)
{
	return 0;
}

__device__ 
void PointLight::setPosition(const Vector& pos)
{
	this->pos = pos;
}

__device__
Vector PointLight::getPosition()
{
	return this->pos;
}

__device__
void PointLight::regulatePower(const float& power)
{
	m_power += power;
}

// Rectangle Light
__device__
RectLight::RectLight()
	: Light()
{
	xSubd = 2;
	ySubd = 2;
	transform.reset(); 

	beginFrame();
}

__device__
RectLight::RectLight(const Vector& translate, const Vector& rotate, const Vector& scale,
					 const Color& color, const float& power, int xSubd, int ySubd)
	: Light(color, power)
	, xSubd(xSubd)
	, ySubd(ySubd)
{
	transform.reset();
	transform.translate(translate);
	transform.rotate(rotate.x, rotate.y, rotate.z);
	transform.scale(scale.x, scale.y, scale.z);

	beginFrame();
}

__device__
void RectLight::beginFrame(void)
{
	center = transform.point(Vector(0, 0, 0));
	Vector a = transform.point(Vector(-0.5, 0.0, -0.5));
	Vector b = transform.point(Vector( 0.5, 0.0, -0.5));
	Vector c = transform.point(Vector( 0.5, 0.0,  0.5));
	float width = (float) (b - a).length();
	float height = (float) (b - c).length();
	area = width * height; // obtain the area of the light, in world space
}

__device__
int RectLight::getNumSamples()
{
	return xSubd * ySubd;
}

__device__
void RectLight::getNthSample(int sampleIdx, const Vector& shadePos, Vector& samplePos, Color& color)
{
	// convert the shade point onto the light's canonic space:
	Vector shadePosCanonical = transform.undoPoint(shadePos);
	
	// shade point "behind" the lamp?
	if (shadePosCanonical.y > 0) {
		color.makeZero();
		return;
	}
	
	// stratified sampling:
	float sx = (sampleIdx % xSubd + randomFloat()) / xSubd;
	float sy = (sampleIdx / xSubd + randomFloat()) / ySubd;
	
	Vector sampleCanonical(sx - 0.5, 0, sy - 0.5);
	samplePos = transform.point(sampleCanonical);
	Vector shadePos_LS = shadePosCanonical - sampleCanonical;
	// return light color, attenuated by the angle of incidence
	// (the cosine between the light's direction and the normed ray toward the hitpos)
	color = this->m_color * (area * this->m_power * float(dot(Vector(0, -1, 0), shadePos_LS) / shadePos_LS.length()));
}

__device__
bool RectLight::intersect(const Ray& ray, double& intersectionDist)
{
	Ray ray_LS = transform.undoRay(ray);
	// check if ray_LS (the incoming ray, transformed in local space) hits the oriented square 1x1, resting
	// at (0, 0, 0), pointing downwards:
	if (ray_LS.start.y >= 0) return false; // ray start is in the wrong subspace; no intersection is possible
	if (ray_LS.dir.y <= 0) return false; // ray direction points downwards; no intersection is possible
	double lengthToIntersection = -(ray_LS.start.y / ray_LS.dir.y); // intersect with XZ plane
	Vector p = ray_LS.start + ray_LS.dir * lengthToIntersection;
	if (fabs(p.x) < 0.5 && fabs(p.z) < 0.5) {
		// the hit point is inside the 1x1 square - calculate the length to the intersection:
		double distance = (transform.point(p) - ray.start).length(); 
		
		if (distance < intersectionDist) {
			intersectionDist = distance;
			return true; // intersection found, and it improves the current closest dist
		}
	}
	return false;
}

__device__
float RectLight::solidAngle(const Vector& x)
{
	Vector x_canonic = transform.undoPoint(x);
	if (x_canonic.y >= 0) return 0;
	Vector x_dir = normalize(x_canonic);
	float cosA = dot(x_dir, Vector(0, -1, 0));
	double d = (x - center).lengthSqr();
	return area * cosA / (1 + d);
}

/// Spot light
__device__
SpotLight::SpotLight(const Vector& position, const Vector& direction, 
					 const Color& color, const float& power,
				     const float& innerAngle, const float& outerAngle)
	: Light(color, power)
	, pos(position)
	, dir(direction)
	, innerAngle(innerAngle)
	, outerAngle(outerAngle)
{
	//dir.normalize();
}

__device__
int SpotLight::getNumSamples()
{
	return 1;
}

__device__
void SpotLight::getNthSample(int sampleIdx, const Vector& shadePos, Vector& samplePos, Color& color)
{
	Vector hitPointDir = pos - shadePos;
	hitPointDir.normalize();

	float cosTheta = dot(dir, hitPointDir);

	float theta = acos(cosTheta);

	if (theta < innerAngle)
	{
		color = m_color * m_power * cosTheta / (shadePos - pos).lengthSqr();
	}

	if (theta > outerAngle)
	{
		color = Color(0, 0, 0);
	}

	if (theta > innerAngle && theta < outerAngle)
	{
		color = m_color * m_power * cosTheta / (shadePos - pos).lengthSqr() / 2;
	}
}

__device__
bool SpotLight::intersect(const Ray& ray, double& intersectionDist)
{
	return false;
}

__device__
float SpotLight::solidAngle(const Vector& x)
{
	return 0;
}