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
	//return Vector(0, 0, -1);
	return m_camera->frontDir;
}

__device__ 
Vector CameraController::backward()
{
	//return Vector(0, 0, 1);
	return -m_camera->frontDir;
}

__device__ 
Vector CameraController::right()
{
	//return Vector(-1, 0, 0);
	return m_camera->rightDir;
}

__device__
Vector CameraController::left()
{
	//return Vector(1, 0, 0);
	return -m_camera->rightDir;
}

__device__ 
void CameraController::moveForward()
{
	//m_camera->pos += forward() * movementSpeed;

	m_camera->pos += m_camera->rightDir * 0;
	m_camera->pos += m_camera->frontDir * movementSpeed;
}

__device__
void CameraController::moveBackward()
{
	//m_camera->pos += backward() * movementSpeed;

	m_camera->pos += m_camera->rightDir * 0;
	m_camera->pos += m_camera->frontDir * -movementSpeed;
}

__device__ 
void CameraController::strafeRight()
{
	//m_camera->pos += right() * movementSpeed;

	m_camera->pos += m_camera->rightDir * movementSpeed;
	m_camera->pos += m_camera->frontDir * 0;
}

__device__ 
void CameraController::strafeLeft()
{
	//m_camera->pos += left() * movementSpeed;

	m_camera->pos += m_camera->rightDir * -movementSpeed;
	m_camera->pos += m_camera->frontDir * 0;
}