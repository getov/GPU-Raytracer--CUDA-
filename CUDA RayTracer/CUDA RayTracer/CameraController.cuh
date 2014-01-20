#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include "Camera.cuh"

class CameraController
{
private:
	Camera* m_camera;
	float movementSpeed;

	// camera directions
	__device__ Vector forward();
	__device__ Vector backward();
	__device__ Vector right();
	__device__ Vector left();

public:
	__device__
	CameraController(Camera& camera, const float& speed);

	// movement
	__device__ void moveForward();
	__device__ void moveBackward();
	__device__ void strafeRight();
	__device__ void strafeLeft();
};

#endif 