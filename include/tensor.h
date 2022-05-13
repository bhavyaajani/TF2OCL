// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include <vector>
#include <stdexcept>
#include <CL\cl.h>

namespace tf2ocl {

	typedef std::vector<int> TensorNDim;

	enum class TensorDType { Float, Double };
	enum class TensorMemType {HOST, DEVICE};

    class Tensor {
	public:
		Tensor(TensorNDim dim, TensorDType dtype);

		virtual ~Tensor();

		TensorNDim ndim()
		{
			return _ndim;
		}

		TensorDType dtype()
		{
			return _dtype;
		}

		size_t count();

		bool allocateHost();

		bool allocateDevice(cl_mem_flags flags);

		void* host()
		{
			return this->_host_ptr;
		}

		cl_mem device()
		{
			return this->_device_ptr;
		}

		bool memcopyHostToDevice();

		bool memcopyDeviceToHost();

		size_t bytes();

		TensorMemType defaultMemType()
		{
			return _defaultMemType;
		}

		void setDefaultMemType(TensorMemType memType)
		{
			_defaultMemType = memType;
		}

	protected:

		void deallocateHost();

		void deallocateDevice();

		TensorDType _dtype;
		TensorNDim _ndim;

		void* _host_ptr;
		cl_mem _device_ptr;

		TensorMemType _defaultMemType;
    };	
}