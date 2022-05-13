// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Flatten.h"
#include <cstring>
#include <iostream>

namespace tf2ocl {

	Flatten::Flatten(const rapidjson::Value &config) : Layer(config)
	{
		if (this->_outputNDim.size() != 1)
		{
			throw std::runtime_error("Excepted Output Tensor NDim is not 1!");
		}
	}

	Flatten::~Flatten()
	{

	}

	bool Flatten::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		const float *ip = (float *) input->host();
		float *op = (float *) _output->host();

		memcpy(op, ip, sizeof(float)*input->count());

		return true;
	}
}