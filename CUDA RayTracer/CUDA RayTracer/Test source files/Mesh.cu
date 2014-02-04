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

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>
#include "Mesh.cuh"
#include "Settings.cuh"
#include "Color.cuh"
#include "AABB.cuh"
#include "Util.cuh"

using std::string;

__device__
void Mesh::initMesh(void)
{
	// calculate a bounding box around the mesh:
	boundingBox.makeEmpty();
	for (int i = 1; i < (int) vertices.size(); i++)
		boundingBox.add(vertices[i]);
	kdroot = NULL;
	if (triangles.size() > 40 && useKDTree) {
		kdroot = new KDTreeNode;
		pgg::vector<int> allTriangles;
		for (int i = 0; i < (int) triangles.size(); i++)
			allTriangles.push_back(i);
		build(*kdroot, boundingBox, allTriangles, 0);
	}
}

__device__
Mesh::~Mesh()
{
	if (kdroot) delete kdroot;
}

//__device__
//const char* Mesh::getName()
//{
//	static char temp[200];
//	sprintf(temp, "Mesh with %d vertices, %d triangles\n", (int) vertices.size(), (int) triangles.size());
//	return temp;
//}

__device__
bool intersectTriangleFast(const Ray& ray, const Vector& A, const Vector& B, const Vector& C, double& dist)
{
	Vector AB = B - A;
	Vector AC = C - A;
	Vector D = -ray.dir;
	//              0               A
	Vector H = ray.start - A;

	/* 2. Solve the equation:
	 *
	 * A + lambda2 * AB + lambda3 * AC = ray.start + gamma * ray.dir
	 *
	 * which can be rearranged as:
	 * lambda2 * AB + lambda3 * AC + gamma * D = ray.start - A
	 *
	 * Which is a linear system of three rows and three unknowns, which we solve using Carmer's rule
	 */

	// Find the determinant of the left part of the equation:
	double Dcr = (AB ^ AC) * D;
	
	// are the ray and triangle parallel?
	if (fabs(Dcr) < 1e-12) return false;
	
	double lambda2 = ( ( H ^ AC) * D ) / Dcr;
	double lambda3 = ( (AB ^  H) * D ) / Dcr;
	double gamma   = ( (AB ^ AC) * H ) / Dcr;

	// is intersection behind us, or too far?
	if (gamma < 0 || gamma > dist) return false;

	// is the intersection outside the triangle?
	if (lambda2 < 0 || lambda2 > 1 || lambda3 < 0 || lambda3 > 1 || lambda2 + lambda3 > 1)
		return false;

	dist = gamma;
	
	
	return true;
}


__device__
bool Mesh::intersectTriangle(const Ray& ray, IntersectionData& data, Triangle& T)
{
	bool inSameDirection = (dot(ray.dir, T.gnormal) > 0);
	//if (backfaceCulling && inSameDirection && !(ray.flags & RF_SHADOW)) return false; // backface culling
	// (backface culling needs to be disabled when we trace shadow rays, otherwise we may find light
	//  in places there shouldn't be one).
	//              B                     A
	Vector AB = vertices[T.v[1]] - vertices[T.v[0]];
	Vector AC = vertices[T.v[2]] - vertices[T.v[0]];
	Vector D = -ray.dir;
	//              0               A
	Vector H = ray.start - vertices[T.v[0]];

	/* 2. Solve the equation:
	 *
	 * A + lambda2 * AB + lambda3 * AC = ray.start + gamma * ray.dir
	 *
	 * which can be rearranged as:
	 * lambda2 * AB + lambda3 * AC + gamma * D = ray.start - A
	 *
	 * Which is a linear system of three rows and three unknowns, which we solve using Carmer's rule
	 */

	// Find the determinant of the left part of the equation:
	double Dcr = (AB ^ AC) * D;
	
	// are the ray and triangle parallel?
	if (fabs(Dcr) < 1e-12) return false;
	
	double rDcr = 1/Dcr;
	
	double lambda2 = ( ( H ^ AC) * D ) * rDcr;
	double lambda3 = ( (AB ^  H) * D ) * rDcr;
	double gamma   = ( (AB ^ AC) * H ) * rDcr;

	// is intersection behind us, or too far?
	if (gamma < 0 || gamma > data.dist) return false;
	
	// is the intersection outside the triangle?
	if (lambda2 < 0 || lambda2 > 1 || lambda3 < 0 || lambda3 > 1 || lambda2 + lambda3 > 1)
		return false;
	//
	
	// intersection found, and it's closer to the current one in data.
	// store intersection point.
	data.p = ray.start + ray.dir * gamma;
	data.dist = gamma;
	data.g = this;
	
	double lambda1 = 1 - lambda2 - lambda3;
	if (faceted || !hasNormals) {
		data.normal = T.gnormal;
	} else {
		// interpolate normals using the barycentric coords:
		data.normal = normals[T.n[0]] * lambda1 +
					  normals[T.n[1]] * lambda2 +
					  normals[T.n[2]] * lambda3;
		data.normal.normalize();
	}
	
	// interpolate the UV texture coords using barycentric coords:
	Vector uv = uvs[T.t[0]] * lambda1 +
				uvs[T.t[1]] * lambda2 +
				uvs[T.t[2]] * lambda3;
	data.u = uv.x;
	data.v = uv.y;
	data.dNdx = T.dNdx;
	data.dNdy = T.dNdy;
	return true;
}

__device__
bool Mesh::intersectKD(KDTreeNode& node, const BBox& bbox, const Ray& ray, IntersectionData& data)
{
	if (node.axis == AXIS_NONE) {
		// leaf node; try intersecting with the triangle list:
		bool found = false;
		for (size_t i = 0; i < node.triangles->size(); i++) {
			int triIdx = (*node.triangles)[i];
			if (intersectTriangle(ray, data, triangles[triIdx])) {
				found = true;
			}
		}
		// the found intersection has to be inside "our" BBox, otherwise we can miss a triangle,
		// as explained in the presentations:
		if (found && bbox.inside(data.p)) return true;
		return false;
	} else {
		// a in-node; intersect with the two children, starting with the closer one first:
		int childOrder[2] = { 0, 1 };
		if (ray.start[node.axis] > node.splitPos)
			dev_swap(childOrder[0], childOrder[1]);
		// 
		BBox childBB[2];
		bbox.split(node.axis, node.splitPos, childBB[0], childBB[1]);
		// name the children bboxen:
		BBox& firstBB = childBB[childOrder[0]];
		BBox& secondBB = childBB[childOrder[1]];
		KDTreeNode& firstChild = node.children[childOrder[0]];
		KDTreeNode& secondChild = node.children[childOrder[1]];
		// if the ray intersects the common wall between the two sub-boxes, then it invariably
		// intersects both boxes (we can skip the testIntersect() checks):
		// (see http://raytracing-bg.net/?q=node/68 )
		if (bbox.intersectWall(node.axis, node.splitPos, ray)) {
			if (intersectKD(firstChild, firstBB, ray, data)) return true;
			return intersectKD(secondChild, secondBB, ray, data);
		} else {
			// if the wall isn't hit, then we intersect exclusively one of the sub-boxes;
			// test one, if the test fails, then it's in the other:
			if (firstBB.testIntersect(ray))
				return intersectKD(firstChild, firstBB, ray, data);
			else
				return intersectKD(secondChild, secondBB, ray, data);
		}
	}
}

__device__
bool Mesh::intersect(Ray ray, IntersectionData& data)
{
	bool found = false;
	// if the ray doesn't intersect the bounding shpere, it is of no use
	// to continue: it can't possibly intersect the mesh.
	if (!boundingBox.testIntersect(ray)) return false;
	
	// if we built a KDTree, use that:
	if (kdroot) {
		return intersectKD(*kdroot, boundingBox, ray, data);
	} else {
		// naive algorithm - iterate and check for intersection all triangles:
		for (size_t i = 0; i < triangles.size(); i++) {
			if (intersectTriangle(ray, data, triangles[i]))
				found = true;
		}
		return found;
	}
}

// parse a string, convert to double. If string is empty, return 0
//__device__
//static double getDouble(const char* s)
//{
//	c_string tmp;
//
//	return static_cast<double>(tmp.atoi(s));
//}
//
//// parse a string, convert to int. If string is empty, return 0
//__device__
//static int getInt(const char* s)
//{
//	c_string tmp;
//	return tmp.atoi(s);
//}

// create a triangle by a OBJ file "f"-line, like "f 1//3 5//3 6//3". The three params in this
// case will be "1//3", "5//3" and "6//3"
//__device__
//Triangle::Triangle(const char* a, const char* b, const char* c)
//{
//	const char* items[3] = { a, b, c };
//	
//	for (int i = 0; i < 3; i++) {
//		const char* item = items[i];
//		
//		vector<char*> subItems = split(item, '/');
//		v[i] = getInt(subItems[0]);
//		if (subItems.size() > 1) {
//			t[i] = getInt(subItems[1]);
//		} else t[i] = 0;
//		if (subItems.size() > 2) {
//			n[i] = getInt(subItems[2]);
//		} else n[i] = 0;
//	}
//}

Triangle::Triangle(std::string a, std::string b, std::string c)
{
	string items[3] = { a, b, c };
	
	for (int i = 0; i < 3; i++) {
		const string& item = items[i];
		
		std::vector<string> subItems = split(item, '/');
		v[i] = getInt(subItems[0]);
		if (subItems.size() > 1) {
			t[i] = getInt(subItems[1]);
		} else t[i] = 0;
		if (subItems.size() > 2) {
			n[i] = getInt(subItems[2]);
		} else n[i] = 0;
	}
}

__device__
void solve2D(double M[2][2], double H[2], double& p, double& q)
{
	// solve a 2x2 linear system:
	// (p, q) * (M) = (H)
	// where p, q are scalars ("unknowns"), M is a 2x2 matrix, and H is a 2-tuple.
	
	double Dcr = M[0][0] * M[1][1] - M[1][0] * M[0][1];
	
	double rDcr = 1 / Dcr;
	
	p = (H[0] * M[1][1] - H[1] * M[0][1]) * rDcr;
	q = (M[0][0] * H[1] - M[1][0] * H[0]) * rDcr;
}


__device__ 
bool Mesh::loadFromOBJ()
{
	//FILE* f = fopen(filename, "rt");

	//if (!f) {
	//	printf("error: no such file: %s", filename);
	//	return false;
	//}
	//
	//vertices.push_back(Vector(0, 0, 0));
	//uvs.push_back(Vector(0, 0, 0));
	//normals.push_back(Vector(0, 0, 0));
	//hasNormals = false;
	//
	//
	//char line[2048];
	//
	//while (fgets(line, sizeof(line), f)) {
	//	if (line[0] == '#') continue;
	//	
	//	vector<char*> tokens = tokenize(line);
	//	
	//	if (tokens.size() == 0) continue;
	//	
	//	// v line - a vertex definition
	//	if (tokens[0] == "v") {
	//		Vector t(getDouble(tokens[1]),
	//		         getDouble(tokens[2]),
	//		         getDouble(tokens[3]));
	//		vertices.push_back(t);
	//		continue;
	//	}

	//	// vn line - a vertex normal definition
	//	if (tokens[0] == "vn") {
	//		hasNormals = true;
	//		Vector t(getDouble(tokens[1]),
	//		         getDouble(tokens[2]),
	//		         getDouble(tokens[3]));
	//		normals.push_back(t);
	//		continue;
	//	}

	//	// vt line - a texture coordinate definition
	//	if (tokens[0] == "vt") {
	//		Vector t(getDouble(tokens[1]),
	//		         getDouble(tokens[2]),
	//		         0);
	//		uvs.push_back(t);
	//		continue;
	//	}
	//	
	//	// f line - a face definition
	//	if (tokens[0] == "f") {
	//		int numTriangles = tokens.size() - 3;
	//		
	//		for (int i = 0; i < numTriangles; i++) {
	//			Triangle T(tokens[1], tokens[2 + i], tokens[3 + i]);
	//			triangles.push_back(T);
	//		}
	//	}
	//}
	
	// preprocess all triangles:
	for (int i = 0; i < (int) triangles.size(); i++) {
		Triangle& T = triangles[i];
		
		Vector AB = vertices[T.v[1]] - vertices[T.v[0]];
		Vector AC = vertices[T.v[2]] - vertices[T.v[0]];
		
		// compute the geometric normal of this triangle:
		T.gnormal = AB ^ AC;
		T.gnormal.normalize();
		
		
		// compute the dNd(x|y) vectors of this triangle:
		double px, py, qx, qy;
		
		Vector AB_2d = uvs[T.t[1]] - uvs[T.t[0]];
		Vector AC_2d = uvs[T.t[2]] - uvs[T.t[0]];
		
		double mat[2][2] = {
			{ AB_2d.x, AC_2d.x },
			{ AB_2d.y, AC_2d.y },
		};
		double h[2] = { 1, 0 };
		
		solve2D(mat, h, px, qx); // (AB_2d * px + AC_2d * qx == (1, 0))
		h[0] = 0; h[1] = 1;
		solve2D(mat, h, py, qy); // (AB_2d * py + AC_2d * qy == (0, 1))
		
		T.dNdx = AB * px + AC * qx;
		T.dNdx.normalize();
		T.dNdy = AB * py + AC * qy;
		T.dNdy.normalize();
	}
	// create the normals[] array - if needed:
	if (!hasNormals && autoSmooth) {
		hasNormals = true;
		normals.resize(vertices.size(), Vector(0, 0, 0)); // extend the normals[] array, and fill with zeros
		for (int i = 0; i < (int) triangles.size(); i++)
			for (int j = 0; j < 3; j++) {
				triangles[i].n[j] = triangles[i].v[j];
				normals[triangles[i].n[j]] += triangles[i].gnormal;
			}
		for (int i = 1; i < (int) normals.size(); i++)
			if (normals[i].lengthSqr() > 1e-9) normals[i].normalize();
	}

	//fclose(f);
	return true;
}

__device__
void Mesh::build(KDTreeNode& node, const BBox& bbox, pgg::vector<int>& tList, int depth)
{
	if (tList.size() < MAX_TRIANGLES_PER_LEAF || depth > MAX_TREE_DEPTH) {
		node.initLeaf(tList);
	} else {
		Axis axis = (Axis) (depth % 3); // alternate splitting planes: X, Y, Z, X, Y, Z, ...
		double axisL = bbox.vmin[axis]; // the left and right extents of the bbox along the chosen axis
		double axisR = bbox.vmax[axis];
		
		// naive split-position choice here: just use the middle of the current bbox.
		// A smarter algo could be used here:
		double splitPos = (axisL + axisR) * 0.5;
		BBox bbLeft, bbRight;
		bbox.split(axis, splitPos, bbLeft, bbRight);
		
		// Split the triangle list into tLeft, tRight, depending on which BBox the triangles
		// intersect with.
		pgg::vector<int> tLeft, tRight;
		for (int i = 0; i < (int) tList.size(); i++) {
			Triangle& T = triangles[tList[i]];
			const Vector& A = vertices[T.v[0]];
			const Vector& B = vertices[T.v[1]];
			const Vector& C = vertices[T.v[2]];
			// usually, a triangle will go either in the left or the right list. In some
			// cases, it may go in both (which is bad, but we hope this would be rare):
			if (bbLeft.intersectTriangle(A, B, C))
				tLeft.push_back(tList[i]);
			if (bbRight.intersectTriangle(A, B, C))
				tRight.push_back(tList[i]);
		}
		node.initBinary(axis, splitPos);
		build(node.children[0], bbLeft, tLeft, depth + 1);
		build(node.children[1], bbRight, tRight, depth + 1);
	}
}
