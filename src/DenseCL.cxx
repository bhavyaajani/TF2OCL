// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\DenseCL.h"
#include "OpenCL\clMatVecMul.h"
#include <iostream>

namespace tf2ocl {

	DenseCL::DenseCL(const rapidjson::Value &config) : Dense(config), LayerCL()
	{
		_operator = new clMatVecMul();
	}

	DenseCL::~DenseCL()
	{
		if (_operator)
			delete _operator;
	}	

	bool DenseCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}
		return _compiled;
	}

	bool DenseCL::operator()(std::vector<Tensor*> inputs)
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

		float *X = (float*) input->host();
		float *M = (float*) this->_kernel->host();
		float *B = (float*)this->_bias->host();

		float *Y = (float *) _output->host();
		
		unsigned int H = this->_kernel->ndim()[0];
		unsigned int W = this->_kernel->ndim()[1];

		_operator->call(M, X, Y, B, H, W);

		if (_activation.compare("relu") == 0)
		{
			for (size_t l = 0; l < _output->ndim()[0]; l++)
			{
				const float v = Y[l];
				Y[l] = (v > 0) ? v : 0;
			}
		}
		else if (_activation.compare("softmax") == 0)
		{
			double norm = 1e-8;

			for (size_t l = 0; l < _output->ndim()[0]; l++)
			{
				norm += std::exp(Y[l]);
			}

			for (size_t l = 0; l < _output->ndim()[0]; l++)
			{
				const float v = Y[l];
				Y[l] = std::exp(v) / norm;
			}
		}
		else if (_activation.compare("sigmoid") == 0)
		{
			for (size_t l = 0; l < _output->ndim()[0]; l++)
			{
				const float v = 1.0/(1.0 + std::exp(-Y[l]));
				Y[l] = v;
			}
		}
		else
		{
		}

		return true;
	}
}