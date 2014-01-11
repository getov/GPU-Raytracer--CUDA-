#include "Sphere.cuh"
#include "Util.cuh"

__device__ bool Sphere::intersect(Ray ray, IntersectionData& data)
{
	// compute the sphere intersection using a quadratic equation:
	Vector H = ray.start - m_center;
	double A = ray.dir.lengthSqr();
	double B = 2 * dot(H, ray.dir);
	double C = H.lengthSqr() - m_radius*m_radius;
	double Dscr = B*B - 4*A*C;
	if (Dscr < 0) return false; // no solutions to the quadratic equation - then we don't have an intersection.
	double x1, x2;
	x1 = (-B + sqrt(Dscr)) / (2*A);
	x2 = (-B - sqrt(Dscr)) / (2*A);
	double sol = x2; // get the closer of the two solutions...
	if (sol < 0) sol = x1; // ... but if it's behind us, opt for the other one
	if (sol < 0) return false; // ... still behind? Then the whole sphere is behind us - no intersection.
	
	// if the distance to the intersection doesn't optimize our current distance, bail out:
	if (sol > data.dist) return false;
	
	data.dist = sol;
	data.p = ray.start + ray.dir * sol;
	data.normal = data.p - m_center; // generate the normal by getting the direction from the center to the ip
	data.normal.normalize();
	data.u = (3.141592653589793238 + atan2(data.p.z - m_center.z, data.p.x - m_center.x))/(2*3.141592653589793238);
	data.v = 1.0 - (3.141592653589793238/2 + asin((data.p.y - m_center.y)/m_radius)) / 3.141592653589793238;
	data.g = this;
	return true;
}