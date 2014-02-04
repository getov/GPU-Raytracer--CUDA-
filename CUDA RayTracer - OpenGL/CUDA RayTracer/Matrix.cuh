#ifndef MATRIX_H
#define MATRIX_H

#include "Vector3D.cuh"

struct Matrix
{
	double m[3][3];
	__device__ Matrix() {}

	__device__ 
	Matrix(const double& diagonalElement)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				if (i == j) m[i][j] = diagonalElement;
				else m[i][j] = 0.0;
	}
};

__device__  
Vector operator * (const Vector& v, const Matrix& m);

__device__  
void operator *= (Vector& v, const Matrix& a);

__device__ 
Matrix operator * (const Matrix& a, const Matrix& b);

__device__ 
double determinant(const Matrix& a);

__device__ 
double cofactor(const Matrix& m, const int& ii, const int& jj);

__device__ 
Matrix inverseMatrix(const Matrix& m);

__device__ 
Matrix rotationAroundX(const double& angle);

__device__ 
Matrix rotationAroundY(const double& angle);

__device__ 
Matrix rotationAroundZ(const double& angle);

#endif // MATRIX_H