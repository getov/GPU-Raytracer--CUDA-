#include "Camera.cuh"

__device__ 
void Camera::beginFrame()
{
	double x = -aspect;
	double y = +1;
	
	Vector corner = Vector(x, y, 1);
	Vector center = Vector(0, 0, 1);
	
	double lenXY = (corner - center).length();
	double wantedLength = tan(toRadians(fov / 2));
	
	double scaling = wantedLength / lenXY;
	
	x *= scaling;
	y *= scaling;

	this->upLeft = Vector(x, y, 1);
	this->upRight = Vector(-x, y, 1);
	this->downLeft = Vector(x, -y, 1);
	
	Matrix rotation = rotationAroundZ(toRadians(roll))
	                * rotationAroundX(toRadians(pitch))
	                * rotationAroundY(toRadians(yaw));
	upLeft *= rotation;
	upRight *= rotation;
	downLeft *= rotation;

	rightDir = Vector(1, 0, 0) * rotation;
	upDir    = Vector(0, 1, 0) * rotation;
	frontDir = Vector(0, 0, 1) * rotation;
	
	upLeft += pos;
	upRight += pos;
	downLeft += pos;
}

__device__ 
Ray Camera::getScreenRay(const double& x, const double& y, int RES_X, int RES_Y)
{
	Ray result; // A, B -     C = A + (B - A) * x
	result.start = this->pos;
	Vector target = upLeft + 
		(upRight - upLeft) * (x / (double) RES_X) +
		(downLeft - upLeft) * (y / (double) RES_Y);
	
	// A - camera; B = target
	result.dir = target - this->pos;
	
	result.dir.normalize();
	
	return result;
}
