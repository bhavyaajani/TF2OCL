// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Concatenate.h"
#include <iostream>

namespace tf2ocl {

	Concatenate::Concatenate(const rapidjson::Value &config) : Layer(config)
	{
		int axis = config["axis"].GetInt();

		if (axis != -1)
		{
			throw new std::runtime_error("Concatenate only supports axis = -1 !");
		}
	}

	Concatenate::~Concatenate()
	{
	}

	Tensor* Concatenate::concate(Tensor *first, Tensor *second)
	{
		const unsigned int count = first->count();

		float *ip1 = (float *)first->host();
		float *ip2 = (float *)second->host();
		float *op = (float *)_output->host();

		const int H = first->ndim()[0];
		const int W = first->ndim()[1];
		const int C1 = first->ndim()[2];
		const int C2 = second->ndim()[2];
		const int C = _output->ndim()[2];

#pragma omp parallel for
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				unsigned int opidx = h*W*C + w*C;
				unsigned int ip1idx = h*W*C1 + w*C1;
				unsigned int ip2idx = h*W*C2 + w*C2;

				for (int c = 0; c < C1; c++)
				{
					op[opidx + c] = ip1[ip1idx + c];
				}

				opidx += C1;

				for (int c = 0; c < C2; c++)
				{
					op[opidx + c] = ip2[ip2idx + c];
				}
			}
		}

		return _output;
	}

	bool Concatenate::operator()(std::vector<Tensor*> inputs)
	{
		if (inputs.size() != 2)
		{
			throw std::runtime_error("Expect 2 inputs!");
		}

		if (this->_inputNDim != inputs[0]->ndim() || inputs[0]->ndim() != inputs[1]->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match !");
		}

		//This is same as class _output!
		concate(inputs[0], inputs[1]);

		return true;
	}
}