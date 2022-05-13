// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\ConcatenateCL.h"
#include "OpenCL\clConcate.h"
#include <iostream>

namespace tf2ocl {

	ConcatenateCL::ConcatenateCL(const rapidjson::Value &config) : Concatenate(config), LayerCL()
	{
		_operator = new clConcate();
	}

	ConcatenateCL::~ConcatenateCL()
	{
		if (_operator)
			delete _operator;
	}

	bool ConcatenateCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}

		return _compiled;
	}

	bool ConcatenateCL::operator()(std::vector<Tensor*> inputs)
	{
		if (inputs.size() != 2)
		{
			throw std::runtime_error("Expect 2 inputs!");
		}

		if (this->_inputNDim != inputs[0]->ndim() || inputs[0]->ndim() != inputs[1]->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match !");
		}

		const int H = inputs[0]->ndim()[0];
		const int W = inputs[0]->ndim()[1];
		const int C1 = inputs[0]->ndim()[2];
		const int C2 = inputs[1]->ndim()[2];
		
		_operator->call(inputs[0]->device(), inputs[1]->device(), _output->device(), H, W, C1, C2);

		return true;
	}
}