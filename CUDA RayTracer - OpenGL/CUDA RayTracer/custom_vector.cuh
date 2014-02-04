#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace pgg
{
	const int INITIAL_CAPACITY = 10;

	template <typename T>
	class vector
	{
	private:
		unsigned int _size;
		unsigned int _space;
		T* _data;

	public:
		__device__ vector();
		__device__ vector(unsigned int s);
		__device__ vector(const vector& obj);

		__device__
		vector& operator=(const vector& obj);

		__device__ ~vector();

		__device__
		void push_back(const T& elem);

		__device__
		unsigned int size();

		__device__
		unsigned int capacity() const;

		__device__
		T& operator[](unsigned int sub);

		__device__ void resize(int newSize, T val = T());
		__device__ void reserve(int newAlloc);
	};

	template <typename T>
	__device__ vector<T>::vector()
		: _size(0)
		, _space(0)
		, _data(nullptr)
	{
	}

	template <typename T>
	__device__ vector<T>::vector(unsigned int s)
		: _size(s)
		, _data(new T[s])
		, _space(s)
	{
		for (int i = 0; i < _size; ++i)
		{
			_data[i] = nullptr;
		}
	}

	template <typename T>
	__device__ void vector<T>::push_back(const T& elem)
	{
		if (_space == 0)
		{
			reserve(INITIAL_CAPACITY);
		}
		else if (_space == _size)
		{
			reserve(2 * _space);
		}

		_data[_size] = elem;
		++_size;
	}

	template <typename T>
	__device__ T& vector<T>::operator[](unsigned int sub)
	{
		if (sub >= 0 || sub < _size)
		{
			return _data[sub];
		}
	}

	template <typename T>
	__device__ unsigned int vector<T>::size()
	{
		return _size;
	}

	template <typename T>
	__device__ unsigned int vector<T>::capacity() const
	{
		return _space;
	}

	template <typename T>
	__device__ void vector<T>::reserve(int newAlloc)
	{
		if (newAlloc <= _space)
		{
			return;
		}

		T* p = new T[newAlloc];
		
		for (int i = 0; i < _size; ++i)
		{
			p[i] = _data[i];
		}
		delete [] _data;

		_data = p;
		_space = newAlloc;
	}

	template <typename T>
	__device__ void vector<T>::resize(int newSize, T val)
	{
		reserve(newSize);

		for (int i = _size; i < newSize; ++i)
		{
			_data[i] = val;
		}
		_size = newSize;
	}

	template <typename T>
	__device__ vector<T>& vector<T>::operator=(const vector& obj)
	{
		if (this == &obj)
		{
			return *this;
		}

		if (obj._size <= _space)
		{
			for (int i = 0; i < obj._size; ++i)
			{
				_data[i] = obj._data[i];
			}
			_size = obj._size;

			return *this;
		}

		T* p = new T[obj._size];

		for (int i = 0; i < obj._size; ++i)
		{
			p[i] = obj._data[i];
		}
		delete [] _data;
		_space = _size = obj._size;
		_data = p;

		return *this;
	}

	template <typename T>
	__device__ vector<T>::~vector()
	{
		delete [] _data;
	}

}