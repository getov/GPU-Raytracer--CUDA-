#ifndef CAMERA_H
#define CAMERA_H

#include "Vector3D.cuh"
#include "Matrix.cuh"
#include "Util.cuh"

class Camera 
{
private:
	Camera(const Camera&);

	// position vectors for the virtual 'screen'
	Vector upLeft, upRight, downLeft;

	// matrix that applies rotation to the Camera
	Matrix rotation;

	/**
	 * @brief - Function that sets the orientation 
	 * and applies rotation to the Camera
	*/
	__device__ void applyOrientation();

	/**
	 * @brief - Function that initializes the virtual 'screen'
	 * on which the image is going to be drawn. We shoot rays
	 * through each pixel of that screen.
	*/
	__device__ void setupScreen();
	
public:
	__device__ Camera();

	__device__
	Camera(double yaw, double pitch, double roll,
		   double fov, double aspectRatio);

	// direction vectors for the Camera
	Vector frontDir, rightDir, upDir;

	Vector pos; // position
	double yaw, pitch, roll; // in degrees
	double fov; // in degrees
	double aspect; 
	
	/**
	 * @brief - Function that sets up the virtual 'screen'
	 * and applies orientation to the camera. 
	 * (calls Camera::setupScreen() and Camera::applyOrientation()
	 * @reference - __device__ void setupScreen()
	 * @reference - __device__ void applyOrientation()
	*/
	__device__ void beginFrame();
	
	/**
	 * @brief - Function that shoots a ray through a pixel on the screen
	 * @param x - the 'x' coordinate of the pixel (horizontal position)
	 * @param y - the 'y' coordinate of the pixel (vertical position)
	 * @param RES_X - the width of the screen
	 * @param RES_Y - the height of the screen
	 * @return Ray - returns the ray shot through the pixel with
	 * cooridates [x , y]
	*/
	__device__ 
	Ray getScreenRay(const double& x, const double& y, int RES_X, int RES_Y);
};


#endif // CAMERA_H
