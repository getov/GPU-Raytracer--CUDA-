#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "Color.cuh"
#include "Vector3D.cuh"
#include "Settings.cuh"
#include "IGeometry.cuh"
#include "Camera.cuh"
#include "IShader.cuh"
#include "Node.cuh"

__device__ 
Color raytrace(Ray ray);

#endif