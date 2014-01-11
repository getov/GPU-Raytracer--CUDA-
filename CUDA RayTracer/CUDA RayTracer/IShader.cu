#include "IShader.cuh"

__device__ 
Shader::Shader(const Color& color)
{
	this->_color = color;
}