#ifndef CAMERA_H
#define CAMERA_H

#include "Vector3D.cuh"
#include "Matrix.cuh"
#include "Util.cuh"
#include "sdl.cuh"

class Camera 
{
	Vector upLeft, upRight, downLeft;
public:
	Vector pos; // position
	double yaw, pitch, roll; // in degrees
	double fov; // in degrees
	double aspect; // 1.3 or ?
	
	__device__ 
	void beginFrame(void);
	
	__device__ 
	Ray getScreenRay(double x, double y);
	
};


#endif // CAMERA_H
