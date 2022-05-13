// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Input.h"
#include <iostream>

namespace tf2ocl {

	Input::Input(const rapidjson::Value &config) : Layer(config)
	{
	}

	Input::~Input()
	{
	}

	bool Input::operator()(std::vector<Tensor*> inputs)
	{
		return true;
	}

	void Input::build_outputs()
	{
		//No Output pass on input as output!
	}

	Tensor* Input::call(std::vector<Tensor*> inputs)
	{
		if (this->_inputNDim != inputs[0]->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}
		return	inputs[0];
	}
}