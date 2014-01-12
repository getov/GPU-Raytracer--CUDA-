#ifndef CONSTANTS_H
#define CONSTANTS_H

//#define FULL_HD

#define VFB_MAX_SIZE 1920

#ifdef FULL_HD
	const int RES_X  = 1920;
	const int RES_Y  = 1080;
#else
	const int RES_X  = 640;
	const int RES_Y  = 480;
#endif

const double PI = 3.141592653589793238;
//#define PI  = 3.141592653589793238; // temporary workaround
#define INF 1e99

const int GEOM_MAX_SIZE = 1024;

__device__ 
static int GEOM_COUNT = 0;

#endif // CONSTANTS_H