#include "CameraController.cuh"

__device__
CameraController::CameraController(Camera& camera, const float& speed)
	: m_camera(&camera)
	, movementSpeed(speed)
{
}

__device__ 
Vector CameraController::forward()
{
	// Vector(0, 0, 0) - Vector(0, 0, 1)   
	return Vector(0, 0, -1);
}

__device__ 
Vector CameraController::backward()
{
	return Vector(0, 0, 1);
}

__device__ 
Vector CameraController::right()
{
	return Vector(-1, 0, 0);
}

__device__
Vector CameraController::left()
{
	return Vector(1, 0, 0);
}

__device__ 
void CameraController::moveForward()
{
	m_camera->pos += forward() * movementSpeed;
}

__device__
void CameraController::moveBackward()
{
	m_camera->pos += backward() * movementSpeed;
}

__device__ 
void CameraController::strafeRight()
{
	m_camera->pos += right() * movementSpeed;
}

__device__ 
void CameraController::strafeLeft()
{
	m_camera->pos = left() * movementSpeed;
}