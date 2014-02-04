/**
* This file contains wrapper function handlers of the 
* CameraController class. They are used to pass the GPU (device)
* code into the CPU (host) code in order to handle the corresponding 
* events in the EventHandler class.
*/

#ifndef RAYTRACER_CONTROLS_H
#define RAYTRACER_CONTROLS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void mvForward();
extern "C" void moveForward();

__global__ void mvBackward();
extern "C" void moveBackward();

__global__ void mvLeft();
extern "C" void strafeLeft();

__global__ void mvRight();
extern "C" void strafeRight();

__global__ void setCamOrientation(float zenith, float azimuth);
extern "C" void setCameraOrientation(float zenith, float azimuth);

#endif