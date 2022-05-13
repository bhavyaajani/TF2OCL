// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Add.h"
#include <iostream>

namespace tf2ocl {

	Add::Add(const rapidjson::Value &config) : Layer(config)
	{
	}

	Add::~Add()
	{
	}

	Tensor* Add::add(Tensor *first, Tensor *second)
	{
		const unsigned int count = first->count();

		float *ip1 = (float *)first->host();
		float *ip2 = (float *)second->host();
		float *op = (float *)_output->host();


#pragma omp parallel for
		for (int i = 0; i < count; i++)
		{
			op[i] = ip1[i] + ip2[i];
		}

		return _output;
	}

	bool Add::operator()(std::vector<Tensor*> inputs)
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
		add(inputs[0], inputs[1]);

		return true;
	}
}