#include "RaytracerControls.cuh"
#include "CameraController.cuh"

extern __device__
CameraController* m_controller;

__global__ void mvForward()
{
	m_controller->moveForward();
}

extern "C" void moveForward()
{
	mvForward<<<1, 1>>>();
}

__global__ void mvBackward()
{
	m_controller->moveBackward();
}

extern "C" void moveBackward()
{
	mvBackward<<<1, 1>>>();
}

__global__ void mvLeft()
{
	m_controller->strafeLeft();
}

extern "C" void strafeLeft()
{
	mvLeft<<<1, 1>>>();
}

__global__ void mvRight()
{
	m_controller->strafeRight();
}

extern "C" void strafeRight()
{
	mvRight<<<1, 1>>>();
}

/// rotation
__global__ 
void setCamOrientation(float zenith, float azimuth)
{
	m_controller->offsetCameraOrientation(zenith, azimuth);
}

extern "C"
void setCameraOrientation(float zenith, float azimuth)
{
	setCamOrientation<<<1, 1>>>(zenith, azimuth);
}