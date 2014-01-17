#ifndef CONSTANTS_H
#define CONSTANTS_H

//#define FULL_HD
#define ANTI_ALIASING
#define VFB_MAX_SIZE 1920
#define INF 1e99
#define MAX_RAY_DEPTH 10

#ifdef FULL_HD
	const int RES_X  = 1920;
	const int RES_Y  = 1080;
#else
	const int RES_X  = 640;
	const int RES_Y  = 480;
#endif

const double PI = 3.141592653589793238;

// the stack size in bytes for each GPU thread
const size_t STACK_SIZE = 5 * 1024;

//#define PI  = 3.141592653589793238; // temporary workaround

const int GEOM_MAX_SIZE = 1024;

__device__ 
static int GEOM_COUNT = 0;

#endif // CONSTANTS_H