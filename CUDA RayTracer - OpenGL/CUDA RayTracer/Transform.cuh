/***************************************************************************
 *   Copyright (C) 2009-2013 by Veselin Georgiev, Slavomir Kaslev et al    *
 *   admin@raytracing-bg.net                                               *
 *                                                                         *
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
#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Vector3D.cuh"
#include "Matrix.cuh"
#include "Util.cuh"

/// A transformation class, which implements model-view transform. Objects can be
/// arbitrarily scaled, rotated and translated.
class Transform
{
private:
	Matrix transform;
	Matrix inverseTransform;
	Vector offset;

public:

	__device__
	Transform() { reset(); }

	__device__
	void reset()
	{
		transform = Matrix(1);
		inverseTransform = inverseMatrix(transform);
		offset.makeZero();
	}

	__device__
	void scale(const double& X, const double& Y, const double& Z) 
	{
		Matrix scaling(X);
		scaling.m[1][1] = Y;
		scaling.m[2][2] = Z;

		transform = transform * scaling;
		inverseTransform = inverseMatrix(scaling) * inverseTransform;
	}

	__device__
	void rotate(const double& yaw, const double& pitch, const double& roll)
	{
		transform = transform *
			rotationAroundX(toRadians(pitch)) *
			rotationAroundY(toRadians(yaw)) *
			rotationAroundZ(toRadians(roll));
		inverseTransform = inverseMatrix(transform);
	}

	__device__
	void translate(const Vector& V) 
	{
		offset = V;
	}

	__device__
	Vector point(Vector P)
	{
		P = P * transform;
		P = P + offset;

		return P;
	}

	__device__
	Vector undoPoint(Vector P) const
	{
		P = P - offset;
		P = P * inverseTransform;

		return P;
	}

	__device__
	Vector direction(const Vector& dir) const
	{
		return dir * transform;
	}

	__device__
	Vector undoDirection(const Vector& dir) const
	{
		return dir * inverseTransform;
	}

	__device__
	Ray undoRay(const Ray& inputRay) const 
	{
		Ray result = inputRay;
		result.start = undoPoint(inputRay.start);
		result.dir   = undoDirection(inputRay.dir);
		return result;
	}
};

#endif // __TRANSFORM_H__

