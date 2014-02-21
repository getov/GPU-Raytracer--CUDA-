/**
* This file contains wrapper function handlers of the 
* CameraController class and other functuins. They are used to pass the GPU (device)
* code into the CPU (host) code in order to handle the corresponding 
* events in the EventHandler class.
*/

#ifndef RAYTRACER_CONTROLS_H
#define RAYTRACER_CONTROLS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// movement
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

// Object transformations
__global__ void target_next_geom();
extern "C" void targetNextGeometry();

__global__ void target_prev_geom();
extern "C" void targetPreviousGeometry();

__global__ void null_current_node();
extern "C" void discardSelectedNode();

__global__ void scale_x(double scaleFactor);
extern "C" void scaleX(double scaleFactor);

__global__ void scale_z(double scaleFactor);
extern "C" void scaleZ(double scaleFactor);

__global__ void scale_y(double scaleFactor);
extern "C" void scaleY(double scaleFactor);

__global__ void rotate_x(double angle);
extern "C" void rotateAroundX(double angle);

__global__ void rotate_y(double angle);
extern "C" void rotateAroundY(double angle);

__global__ void rotate_z(double angle);
extern "C" void rotateAroundZ(double angle);

__global__ void translate_x(double translateFactor);
extern "C" void translateX(double translateFactor);

__global__ void translate_y(double translateFactor);
extern "C" void translateY(double translateFactor);

__global__ void translate_z(double translateFactor);
extern "C" void translateZ(double translateFactor);

/**
 * Light Movement
 * Only 3 functions for representing all 
 * the 6 directions towards the axes
 * (x, -x, z, -z, y, -y)
 * @param dir - if positive, move towards the positive direction
 *				if negative, move towards the negative direction
*/ 
__global__ void move_light_x(int dir);
extern "C" void moveLightX(int dir);

__global__ void move_light_z(int dir);
extern "C" void moveLightZ(int dir);

__global__ void move_light_y(int dir);
extern "C" void moveLightY(int dir);

__global__ void reg_light_power(int mult);
extern "C" void regulateLightPower(int mult);

// Shader selection
__global__ void get_next_shader();
extern "C" void getNextShader();

__global__ void get_prev_shader();
extern "C" void getPreviousShader();

// generate geometry
__global__ void gen_sphere();
extern "C" void generateSphere();

#endif