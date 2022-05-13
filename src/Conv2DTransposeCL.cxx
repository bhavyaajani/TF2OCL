// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Conv2DTransposeCL.h"
#include "OpenCL\clConv2DTranspose.h"

#include <iostream>

namespace tf2ocl {

	Conv2DTransposeCL::Conv2DTransposeCL(const rapidjson::Value &config) : Conv2DTranspose(config), LayerCL()
	{
		_operator = new clConv2DTranspose();
	}

	Conv2DTransposeCL::~Conv2DTransposeCL()
	{
		if (_operator)
			delete _operator;
	}

	bool Conv2DTransposeCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}

		if (_compiled)
		{
			_compiled &= this->_kernel->allocateDevice(CL_MEM_READ_ONLY);
			_compiled &= this->_kernel->memcopyHostToDevice();

			if (this->_bias)
			{
				_compiled &= this->_bias->allocateDevice(CL_MEM_READ_ONLY);
				_compiled &= this->_bias->memcopyHostToDevice();
			}
		}

		return _compiled;
	}

	bool Conv2DTransposeCL::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		if (!_compiled)
		{
			throw std::runtime_error("OpenCL layer not compiled yet !");
		}

		unsigned int iH = input->ndim()[0];
		unsigned int iW = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		unsigned int oH = _output->ndim()[0];
		unsigned int oW = _output->ndim()[1];
		unsigned int C = _output->ndim()[2];

		unsigned int applyReLU = (_activation.compare("relu") == 0);
		unsigned int applySigmoid = (_activation.compare("sigmoid") == 0);

		cl_mem bias = (_bias) ? _bias->device() : 0;
		_operator->call(input->device(), _kernel->device(), bias, _output->device(),
			iH, iW, D, oH, oW, C, applyReLU);

		if (applySigmoid)
		{
			_output->allocateHost();
			_output->memcopyDeviceToHost();

			float *Y = (float *)_output->host();
			const unsigned int count = oH*oW*C;

#pragma omp parallel for
			for (int index = 0; index < count; index++)
			{
				Y[index] = 1.0 / (1.0 + std::exp(-Y[index]));
			}

			_output->setDefaultMemType(TensorMemType::HOST);
		}

		return true;
	}
}