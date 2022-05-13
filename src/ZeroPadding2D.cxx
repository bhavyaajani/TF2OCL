// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\ZeroPadding2D.h"
#include <iostream>

namespace tf2ocl {

	ZeroPadding2D::ZeroPadding2D(const rapidjson::Value &config) : Layer(config)
	{
		const rapidjson::Value &padding = config["padding"];
		const rapidjson::Value &padHeight = padding[0];
		const rapidjson::Value &padWidth = padding[1];

		_padding.push_back(padHeight[0].GetUint());
		_padding.push_back(padHeight[1].GetUint());
		_padding.push_back(padWidth[0].GetUint());
		_padding.push_back(padWidth[1].GetUint());
	}

	ZeroPadding2D::~ZeroPadding2D()
	{
	}

	Tensor* ZeroPadding2D::padding(Tensor *input)
	{
		unsigned int iH = input->ndim()[0];
		unsigned int iW = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		unsigned int oH = _output->ndim()[0];
		unsigned int oW = _output->ndim()[1];

		float *ip = (float *)input->host();
		float *op = (float *)_output->host();

		unsigned int count = iW*D*sizeof(float);

#pragma omp parallel for
		for (int h = 0; h < iH; h++)
		{
			float *src = ip + h*iW*D;
			float *dst = op + (h + _padding[0])*(oW*D) + _padding[2]*D;

			std::memcpy(dst, src, count);
		}

		return _output;
	}

	bool ZeroPadding2D::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		//This is same as class _output!
		padding(input);
		return true;
	}
}