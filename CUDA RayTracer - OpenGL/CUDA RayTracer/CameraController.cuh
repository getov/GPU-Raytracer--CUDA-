#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include "Camera.cuh"

class CameraController
{
private:
	Camera* m_camera;
	float movementSpeed;

	/**
	 * @brief - Wrapper function for returning
	 * direction towards the positive z-axis
	 * @return - returns Camera::frontDir, that is a
	 * direction vector towards the z-axis
	*/
	__device__ Vector forward();

	/**
	 * @brief - Wrapper function for returning
	 * direction towards the negative z-axis
	 * @return - returns -Camera::frontDir, that is a direction
	 * vector (the opposite of Camera::frontDir), towards the z-axis
	*/
	__device__ Vector backward();

	/**
	 * @brief - Wrapper function for returning
	 * direction towards the positive x-axis
	 * @return - returns Camera::rightDir, that is a
	 * direction vector towards the x-axis
	*/
	__device__ Vector right();

	/**
	 * @brief - Wrapper function for returning
	 * direction towards the negative x-axis
	 * @return - returns -Camera::rightDir, that is a direction
	 * vector (the opposite of Camera::rightDir), towards the x-axis
	*/
	__device__ Vector left();
	 
public:
	__device__
	CameraController(Camera& camera, const float& speed);

	/**
	 * @brief - 	Moves the Camera position 
	 * towards the positive z-axis. The speed of
	 * the motion is determined by CameraController::movementSpeed
	*/
	__device__ void moveForward();

	/**
	 * @brief - 	Moves the Camera position 
	 * towards the negative z-axis. The speed of
	 * the motion is determined by CameraController::movementSpeed
	*/
	__device__ void moveBackward();

	/**
	 * @brief - 	Moves the Camera position 
	 * towards the positive x-axis. The speed of
	 * the motion is determined by CameraController::movementSpeed
	*/
	__device__ void strafeRight();

	/**
	 * @brief - 	Moves the Camera position 
	 * towards the negative x-axis. The speed of
	 * the motion is determined by CameraController::movementSpeed
	*/
	__device__ void strafeLeft();

	/**
	 * @brief - Handles the Camera rotation to all directions
	 * @param zenith - the vertical angle movement in degrees
	 * @param azimuth - the horizontal angle movement in degrees
	*/
	__device__
	void offsetCameraOrientation(const float& zenith, const float& azimuth);
};

#endif 