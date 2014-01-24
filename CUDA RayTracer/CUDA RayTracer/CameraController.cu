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
	return m_camera->frontDir;
}

__device__ 
Vector CameraController::backward()
{
	return -m_camera->frontDir;
}

__device__ 
Vector CameraController::right()
{
	return m_camera->rightDir;
}

__device__
Vector CameraController::left()
{
	return -m_camera->rightDir;
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
	m_camera->pos += left() * movementSpeed;
}

// rotation
__device__                         
void CameraController::offsetCameraOrientation(const float& zenith, const float& azimuth)
{
	m_camera->yaw += -azimuth;

	while (m_camera->yaw > 360.0f)
	{
		m_camera->yaw -= 360.0;
	}
	while (m_camera->yaw < 0.0f)
	{
		m_camera->yaw += 360.0;
	}

	// to prevent gimbal lock
	m_camera->pitch += -zenith;
	if (m_camera->pitch > 85.0f)
	{
		m_camera->pitch = 85.0f;
	}
	if (m_camera->pitch < -85.0f)
	{
		m_camera->pitch = -85.0f;
	}
}