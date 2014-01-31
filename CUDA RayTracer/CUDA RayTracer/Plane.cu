#include "Plane.cuh"

__device__
Plane::Plane(double y, double limitX, double limitZ)
	 : _y(y) 
	 , _limitX(limitX)
	 , _limitZ(limitZ)
{
}

__device__
bool Plane::intersect(Ray ray, IntersectionData& data)
{
	// intersect a ray with a XZ plane:
	// if the ray is pointing to the horizon, or "up", but the plane is below us,
	// of if the ray is pointing down, and the plane is above us, we have no intersection
	
	if ((ray.start.y > _y && ray.dir.y > -1e-9) || (ray.start.y < _y && ray.dir.y < 1e-9))
	{
		return false;
	}
	else
	{
		double yDiff = ray.dir.y;
		double wantYDiff = ray.start.y - this->_y;
		double mult = wantYDiff / -yDiff;
		
		// if the distance to the intersection (mult) doesn't optimize our current distance, bail out:
		if (mult > data.dist)
		{
			return false;
		}
		
		Vector crossPoint = ray.start + ray.dir * mult;
		if (fabs(crossPoint.x) > _limitX || fabs(crossPoint.z) > _limitZ)
		{
			return false;
		}

		// calculate intersection:
		data.p = ray.start + ray.dir * mult;
		data.dist = mult;
		data.normal = Vector(0, 1, 0);
		data.dNdx = Vector(1, 0, 0);
		data.dNdy = Vector(0, 0, 1);
		data.u = data.p.x;
		data.v = data.p.z;
		data.g = this;
		data.geomID = GeometryID::PLANE;

		return true;
	}
}