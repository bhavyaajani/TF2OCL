// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\ReLU.h"
#include <iostream>

namespace tf2ocl {

	ReLU::ReLU(const rapidjson::Value &config) : Layer(config)
	{
		
	}

	ReLU::~ReLU()
	{
	}

	Tensor* ReLU::relu(Tensor *input)
	{
		const unsigned int count = input->count();

		float *ip = (float *)input->host();
		float *op = (float *)_output->host();


#pragma omp parallel for
		for (int i = 0; i < count; i++)
		{
			const float v = ip[i];
			op[i] = (v > 0) ? v : 0;
		}

		return _output;
	}

	bool ReLU::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		//This is same as class _output!
		relu(input);

		return true;
	}
}