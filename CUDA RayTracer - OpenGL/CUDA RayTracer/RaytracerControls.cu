#include "RaytracerControls.cuh"
#include "CameraController.cuh"

extern __device__
CameraController* controller;

// movement
__global__ void mvForward()
{
	controller->moveForward();
}
extern "C" void moveForward()
{
	mvForward<<<1, 1>>>();
}

__global__ void mvBackward()
{
	controller->moveBackward();
}
extern "C" void moveBackward()
{
	mvBackward<<<1, 1>>>();
}

__global__ void mvLeft()
{
	controller->strafeLeft();
}
extern "C" void strafeLeft()
{
	mvLeft<<<1, 1>>>();
}

__global__ void mvRight()
{
	controller->strafeRight();
}
extern "C" void strafeRight()
{
	mvRight<<<1, 1>>>();
}

// rotation
__global__ 
void setCamOrientation(float zenith, float azimuth)
{
	controller->offsetCameraOrientation(zenith, azimuth);
}

extern "C"
void setCameraOrientation(float zenith, float azimuth)
{
	setCamOrientation<<<1, 1>>>(zenith, azimuth);
}