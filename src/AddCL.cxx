// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\AddCL.h"
#include "OpenCL\clAdd.h"
#include <iostream>

namespace tf2ocl {

	AddCL::AddCL(const rapidjson::Value &config) : Add(config), LayerCL()
	{
		_operator = new clAdd();
	}

	AddCL::~AddCL()
	{
		if (_operator)
			delete _operator;
	}

	bool AddCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}

		return _compiled;
	}

	bool AddCL::operator()(std::vector<Tensor*> inputs)
	{
		if (inputs.size() != 2)
		{
			throw std::runtime_error("Expect 2 inputs!");
		}

		if (this->_inputNDim != inputs[0]->ndim() || inputs[0]->ndim() != inputs[1]->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match !");
		}

		unsigned int count = _output->count();
		_operator->call(inputs[0]->device(), inputs[1]->device(), _output->device(), count);

		return true;
	}
}