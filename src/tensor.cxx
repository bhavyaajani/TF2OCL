// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Tensor.h"
#include <OpenCL\clContext.h>

namespace tf2ocl {

	Tensor::Tensor(TensorNDim dim, TensorDType dtype)
	{
		this->_ndim.clear();
		this->_dtype = dtype;
		this->_host_ptr = NULL;
		this->_device_ptr = 0;

		this->_defaultMemType = TensorMemType::HOST;

		if (dim.empty())
		{
			throw std::runtime_error(std::string("Can not instantiate a Tensor with 0 NDim"));
		}

		for (int d = 0; d<dim.size(); d++)
		{
			this->_ndim.push_back(dim[d]);
		}
	}

	Tensor::~Tensor()
	{
		this->deallocateHost();
		this->deallocateDevice();
	}

	size_t Tensor::count()
	{
		if (this->_ndim.size() == 0)
			return 0;

		std::size_t size = 1;
		for (size_t s = 0; s < this->_ndim.size(); s++)
		{
			size *= this->_ndim[s];
		}

		return size;
	}

	bool Tensor::allocateHost()
	{
		if (_host_ptr == NULL)
		{
			size_t size = count();

			switch (this->_dtype)
			{
			case TensorDType::Float:
				_host_ptr = new float[size] {0};
				break;
			case TensorDType::Double:
				_host_ptr = new double[size] {0};
				break;
			default:
				throw std::runtime_error(std::string("Unsupported dtype encountered!"));
			}
		}

		return (_host_ptr != NULL);
	}

	void Tensor::deallocateHost()
	{
		if (_host_ptr)
		{
			switch (this->_dtype)
			{
			case TensorDType::Float:
				delete[](float*) (_host_ptr);
				break;
			case TensorDType::Double:
				delete[](double*) (_host_ptr);
				break;
			default:
				throw std::runtime_error(std::string("Unsupported dtype encountered!"));
			}
		}

		_host_ptr = NULL;
	}

	bool Tensor::allocateDevice(cl_mem_flags flags)
	{
		if (_device_ptr == 0)
		{
			size_t size = count();
			size_t bytes;

			switch (this->_dtype)
			{
			case TensorDType::Float:
				bytes = size*sizeof(float);
				break;
			case TensorDType::Double:
				bytes = size*sizeof(double);
				break;
			default:
				throw std::runtime_error(std::string("Unsupported dtype encountered!"));
			}

			cl_int err;
			cl_context context = CLContext::Instance()->getContext();
			_device_ptr = clCreateBuffer(context, flags, bytes, NULL, &err);
		}

		return (_device_ptr != 0);
	}

	void Tensor::deallocateDevice()
	{
		if (_device_ptr)
		{
			clReleaseMemObject(_device_ptr);
			_device_ptr = 0;
		}
	}

	bool Tensor::memcopyHostToDevice()
	{
		if (_host_ptr && _device_ptr)
		{
			cl_context oclcontext = CLContext::Instance()->getContext();
			cl_device_id deviceId = CLContext::Instance()->getDeviceId();

			cl_int err;
			// Create a command queue
			cl_command_queue queue = clCreateCommandQueue(oclcontext, deviceId, 0, &err);

			size_t bytes = this->bytes();
			// Write our data set into the input array in device memory
			err |= clEnqueueWriteBuffer(queue, _device_ptr, CL_TRUE, 0, bytes, _host_ptr, 0, NULL, NULL);
			clReleaseCommandQueue(queue);
		}
		else
		{
			throw new std::runtime_error("First allocate host/device memory!");
		}
		//TBD
		return true;
	}

	bool Tensor::memcopyDeviceToHost()
	{
		if (_device_ptr && _host_ptr)
		{
			cl_context oclcontext = CLContext::Instance()->getContext();
			cl_device_id deviceId = CLContext::Instance()->getDeviceId();

			cl_int err;
			// Create a command queue
			cl_command_queue queue = clCreateCommandQueue(oclcontext, deviceId, 0, &err);

			size_t bytes = this->bytes();
			// Read the results from the device
			err = clEnqueueReadBuffer(queue, _device_ptr, CL_TRUE, 0, bytes, _host_ptr, 0, NULL, NULL);
			clReleaseCommandQueue(queue);
		}
		else
		{
			throw new std::runtime_error("First allocate device/host memory!");
		}

		//TBD
		return true;
	}

	size_t Tensor::bytes()
	{
		size_t size = count();
		size_t bytes;

		switch (this->_dtype)
		{
		case TensorDType::Float:
			bytes = size*sizeof(float);
			break;
		case TensorDType::Double:
			bytes = size*sizeof(double);
			break;
		default:
			throw std::runtime_error(std::string("Unsupported dtype encountered!"));
		}

		return bytes;
	}
}