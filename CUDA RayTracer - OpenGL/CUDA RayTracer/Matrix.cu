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

#include "Matrix.cuh"

__device__  
void operator *= (Vector& v, const Matrix& a)
{
	v = v*a; 
}

__device__  
Vector operator * (const Vector& v, const Matrix& m)
{
	return Vector(
		v.x * m.m[0][0] + v.y * m.m[1][0] + v.z * m.m[2][0],
		v.x * m.m[0][1] + v.y * m.m[1][1] + v.z * m.m[2][1],
		v.x * m.m[0][2] + v.y * m.m[1][2] + v.z * m.m[2][2]
	);
}

//!< matrix multiplication; result = a*b
__device__ 
Matrix operator * (const Matrix& a, const Matrix& b)
{
	Matrix c(0.0);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				c.m[i][j] += a.m[i][k] * b.m[k][j];
	return c;
}

//!< finds the determinant of a matrix
__device__ 
double determinant(const Matrix& a)
{
	return a.m[0][0] * a.m[1][1] * a.m[2][2]
	     - a.m[0][0] * a.m[1][2] * a.m[2][1]
	     - a.m[0][1] * a.m[1][0] * a.m[2][2]
	     + a.m[0][1] * a.m[1][2] * a.m[2][0]
	     + a.m[0][2] * a.m[1][0] * a.m[2][1]
	     - a.m[0][2] * a.m[1][1] * a.m[2][0];
}

__device__ 
double cofactor(const Matrix& m, const int& ii, const int& jj)
{
	int rows[2], rc = 0, cols[2], cc = 0;
	for (int i = 0; i < 3; i++)
		if (i != ii) rows[rc++] = i;
	for (int j = 0; j < 3; j++)
		if (j != jj) cols[cc++] = j;
	double t = m.m[rows[0]][cols[0]] * m.m[rows[1]][cols[1]] - m.m[rows[1]][cols[0]] * m.m[rows[0]][cols[1]];
	if ((ii + jj) % 2) t = -t;
	return t;
}

//!< finds the inverse of a matrix (assuming it exists)
__device__ 
Matrix inverseMatrix(const Matrix& m)
{
	double D = determinant(m);
	if (fabs(D) < 1e-12) return m; // an error; matrix is not invertible
	double rD = 1.0 / D;
	Matrix result;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			result.m[i][j] = rD * cofactor(m, j, i);
	return result;
}

//!< returns a rotation matrix around the X axis; the angle is in radians
__device__ 
Matrix rotationAroundX(const double& angle)
{
	double S = sin(angle);
	double C = cos(angle);
	Matrix a(1.0);
	a.m[1][1] = C;
	a.m[2][1] = S;
	a.m[1][2] = -S;
	a.m[2][2] = C;
	return a;
}

//!< same as above, but rotate around Y
__device__
Matrix rotationAroundY(const double& angle)
{
	double S = sin(angle);
	double C = cos(angle);
	Matrix a(1.0);
	a.m[0][0] = C;
	a.m[2][0] = -S;
	a.m[0][2] = S;
	a.m[2][2] = C;
	return a;
}

//!< same as above, but rotate around Z
__device__ 
Matrix rotationAroundZ(const double& angle)
{
	double S = sin(angle);
	double C = cos(angle);
	Matrix a(1.0);
	a.m[0][0] = C;
	a.m[1][0] = S;
	a.m[0][1] = -S;
	a.m[1][1] = C;
	return a;
}