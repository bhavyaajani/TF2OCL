// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\MaxPooling2D.h"
#include <iostream>

namespace tf2ocl {

	MaxPooling2D::MaxPooling2D(const rapidjson::Value &config) : Layer(config)
	{
		const rapidjson::Value & pooling = config["pool_size"];
		_pool_size.push_back(pooling[0].GetInt());
		_pool_size.push_back(pooling[1].GetInt());
		
		const rapidjson::Value & strides = config["strides"];
		_strides.push_back(strides[0].GetInt());
		_strides.push_back(strides[1].GetInt());

		std::string padding = config["padding"].GetString();
		if (padding.compare("valid") != 0)
		{
			std::cout << "Invalid padding scheme!" << std::endl;
			throw new std::runtime_error("Invalid Padding scheme!");
		}
	}

	MaxPooling2D::~MaxPooling2D()
	{
	}

	void MaxPooling2D::max2D(Tensor *input, float *out, unsigned int i, unsigned j, int ph, int pw)
	{
		unsigned int H = input->ndim()[0];
		unsigned int W = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		const float *in = (float*)input->host();

		for (int d = 0; d < D; d++)
		{
			float maximum = in[i*W*D + j*D + d];

			for (int h = i; h < i + ph; h++)
			{
				for (int w = j; w < j + pw; w++)
				{
					float v = in[h*W*D + w*D + d];
					if (v > maximum) {
						maximum = v;
					}
				}
			}

			out[d] = maximum;
		}
	}

	Tensor* MaxPooling2D::maxPool2D(Tensor *input, int ph, int pw, int sh, int sw)
	{
		unsigned int H = input->ndim()[0];
		unsigned int W = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		unsigned int oH = _output->ndim()[0];
		unsigned int oW = _output->ndim()[1];

		float *op = (float *)_output->host();

#pragma omp parallel for
		for (int h = 0; h < oH; h++)
		{
			for (int w = 0; w < oW; w++)
			{
				float *buffer = op + h*oW*D + w*D;
				unsigned int i = h*sh;
				unsigned int j = w*sw;
				max2D(input, buffer, i, j, ph, pw);
			}
		}

		return _output;
	}

	bool MaxPooling2D::operator()(std::vector<Tensor*> inputs)
	{
		Tensor* input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		//This is same as class _output!
		maxPool2D(input, _pool_size[0], _pool_size[1], _strides[0], _strides[1]);
		return true;
	}
}