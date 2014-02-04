#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "Color.cuh"
#include "Vector3D.cuh"
#include "Settings.cuh"
#include "IGeometry.cuh"
#include "Camera.cuh"
#include "IShader.cuh"
#include "Node.cuh"
#include "custom_vector.cuh"
#include "CameraController.cuh"
#include "Scene.cuh"

class Scene;

extern __device__ Scene* scene;

__device__ 
Color raytrace(Ray ray);

__device__
bool testVisibility(const Vector& from, const Vector& to);


#endif