// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\GlobalAvgPooling2D.h"
#include <iostream>

namespace tf2ocl {

	GlobalAvgPooling2D::GlobalAvgPooling2D(const rapidjson::Value &config) : Layer(config)
	{
	}

	GlobalAvgPooling2D::~GlobalAvgPooling2D()
	{
	}

	Tensor* GlobalAvgPooling2D::globalAvg2D(Tensor *input)
	{
		unsigned int H = input->ndim()[0];
		unsigned int W = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		const unsigned int count = W*H;

		float *ip = (float *)input->host();
		float *op = (float *)_output->host();

#pragma omp parallel for
		for (int d = 0; d < D; d++)
		{
			float avg = 0;

			for (int i = 0; i < count; i++)
			{
				avg += ip[i*D + d];
			}
			
			op[d] = avg / count;
		}

		return _output;
	}

	bool GlobalAvgPooling2D::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		//This is same as class _output!
		globalAvg2D(input);
		return true;
	}
}