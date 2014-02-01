#ifndef SETTINGS_H
#define SETTINGS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define FULL_HD
#define REAL_TIME_RENDERING
#define VFB_MAX_SIZE 1920
#define INF 1e99
#define MAX_RAY_DEPTH 5
#define PI 3.141592653589793238

#define MAX_TRIANGLES_PER_LEAF 20
#define MAX_TREE_DEPTH         64

struct GlobalSettings
{
	static bool AAEnabled;
	static bool previewAA;
	static bool realTime;
	static int RES_X;
	static int RES_Y;
};

// the stack size in bytes for each GPU thread
const size_t STACK_SIZE = 20 * 1024;

#ifdef FULL_HD
	const int RES_X  = 1920;
	const int RES_Y  = 1080;
#else
	const int RES_X  = 640;
	const int RES_Y  = 480;
#endif


const int GEOM_MAX_SIZE = 1024;

__device__ 
static int GEOM_COUNT = 0;

#endif // SETTINGS_H