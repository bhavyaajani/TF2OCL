// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\ReLUCL.h"
#include "OpenCL\clReLU.h"
#include <iostream>

namespace tf2ocl {

	ReLUCL::ReLUCL(const rapidjson::Value &config) : ReLU(config), LayerCL()
	{
		_operator = new clReLU();
	}

	ReLUCL::~ReLUCL()
	{
		if (_operator)
			delete _operator;
	}

	bool ReLUCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}

		return _compiled;
	}

	bool ReLUCL::operator()(std::vector<Tensor*> inputs)
	{
		if (inputs.size() != 1)
		{
			throw std::runtime_error("Expect 1 inputs!");
		}

		unsigned int count = _output->count();
		_operator->call(inputs[0]->device(),_output->device(),count);

		return true;
	}
}