#pragma once

#include "Vector3D.cuh"
#include "AABB.cuh"
#include <vector>
#include <string>
using std::string;
using std::vector;

class ObjectLoader
{
public:
	vector<Vector> vertices; //!< An array with all vertices in the mesh
	vector<Vector> normals; //!< An array with all normals in the mesh
	vector<Vector> uvs; //!< An array with all texture coordinates in the mesh
	vector<Triangle> triangles; //!< An array that holds all triangles

	bool loadFromOBJ(const char* fileName);
};