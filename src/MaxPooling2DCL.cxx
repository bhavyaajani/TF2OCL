// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\MaxPooling2DCL.h"
#include "OpenCL\clMaxPool2D.h"
#include <iostream>

namespace tf2ocl {

	MaxPooling2DCL::MaxPooling2DCL(const rapidjson::Value &config) : MaxPooling2D(config), LayerCL()
	{
		_operator = new clMaxPool2D();
	}

	MaxPooling2DCL::~MaxPooling2DCL()
	{
		if (_operator)
			delete _operator;
	}

	bool MaxPooling2DCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}

		return _compiled;
	}

	bool MaxPooling2DCL::operator()(std::vector<Tensor*> inputs)
	{
		if (inputs.size() != 1)
		{
			throw std::runtime_error("Expect 1 inputs!");
		}

		unsigned int iH = inputs[0]->ndim()[0];
		unsigned int iW = inputs[0]->ndim()[1];
		unsigned int D =  inputs[0]->ndim()[2];

		unsigned int oH = _output->ndim()[0];
		unsigned int oW = _output->ndim()[1];

		_operator->call(inputs[0]->device(), _output->device(), iH, iW, D, 
			oH, oW, _pool_size[0], _pool_size[1], _strides[0], _strides[1]);

		return true;
	}
}