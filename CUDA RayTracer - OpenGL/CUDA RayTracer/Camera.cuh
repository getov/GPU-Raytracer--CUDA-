/***************************************************************************
 *   Copyright (C) 2009-2013 by Veselin Georgiev, Slavomir Kaslev et al    *
 *   admin@raytracing-bg.net                                               *
 *																		   *
 *	 Contributor: Peter Getov											   *
 *																		   *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef CAMERA_H
#define CAMERA_H

#include "Vector3D.cuh"
#include "Matrix.cuh"
#include "Util.cuh"

class Camera 
{
private:
	Vector upLeft, upRight, downLeft;
	//Matrix rotation;

	__device__
	void applyOrientation();
	
public:
	Vector frontDir, rightDir, upDir;

	Vector pos; // position
	double yaw, pitch, roll; // in degrees
	double fov; // in degrees
	double aspect; 
	
	__device__ 
	void beginFrame();
	
	__device__ 
	Ray getScreenRay(const double& x, const double& y, int RES_X, int RES_Y);
};


#endif // CAMERA_H
