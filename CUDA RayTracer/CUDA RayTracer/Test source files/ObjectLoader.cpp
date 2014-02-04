#include "ObjectLoader.h"
#include "Util.cuh"


bool ObjectLoader::loadFromOBJ(const char* fileName)
{
	FILE* f = fopen(fileName, "rt");
	
	if (!f) {
		printf("error: no such file: %s", fileName);
		return false;
	}
	
	vertices.push_back(Vector(0, 0, 0));
	uvs.push_back(Vector(0, 0, 0));
	normals.push_back(Vector(0, 0, 0));
	GlobalSettings::hasNormals = false;
	
	
	char line[2048];
	
	while (fgets(line, sizeof(line), f)) {
		if (line[0] == '#') continue;
		
		vector<string> tokens = tokenize(string(line));
		
		if (tokens.empty()) continue;
		
		// v line - a vertex definition
		if (tokens[0] == "v") {
			Vector t(getDouble(tokens[1]),
			         getDouble(tokens[2]),
			         getDouble(tokens[3]));
			vertices.push_back(t);
			continue;
		}

		// vn line - a vertex normal definition
		if (tokens[0] == "vn") {
			GlobalSettings::hasNormals = true;
			Vector t(getDouble(tokens[1]),
			         getDouble(tokens[2]),
			         getDouble(tokens[3]));
			normals.push_back(t);
			continue;
		}

		// vt line - a texture coordinate definition
		if (tokens[0] == "vt") {
			Vector t(getDouble(tokens[1]),
			         getDouble(tokens[2]),
			         0);
			uvs.push_back(t);
			continue;
		}
		
		// f line - a face definition
		if (tokens[0] == "f") {
			int numTriangles = tokens.size() - 3;
			
			for (int i = 0; i < numTriangles; i++) {
				Triangle T(tokens[1], tokens[2 + i], tokens[3 + i]);
				triangles.push_back(T);
			}
		}
	}
	fclose(f);
}

