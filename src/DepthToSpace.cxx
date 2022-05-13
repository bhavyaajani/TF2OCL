// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\DepthToSpace.h"
#include <iostream>

namespace tf2ocl {

	DepthToSpace::DepthToSpace(const rapidjson::Value &config) : Layer(config)
	{
	}

	DepthToSpace::~DepthToSpace()
	{
	}

	Tensor* DepthToSpace::depth2space(Tensor *input)
	{
		const float *ip = (float *)input->host();
		float *op = (float *)_output->host();

		const int iH = input->ndim()[0];
		const int iW = input->ndim()[1];
		const int iD = input->ndim()[2];
		
		const int oH = _output->ndim()[0];
		const int oW = _output->ndim()[1];
		const int oD = _output->ndim()[2];

		if ((oH / iH) != (oW / iW))
		{
			throw std::runtime_error("Block size should be symmetrical along H and W axis !");
		}

		const int Block = oH / iH;

		if (oD*Block*Block != iD)
		{
			throw std::runtime_error("Input / Output depth does not seems to be correct for DepthToSpace operation!");
		}
		
		if (oD != 1)
		{
			throw std::runtime_error("Only support Output depth == 1 !");
		}

#pragma omp parallel for
		for (int h = 0; h < oH; h++)
		{
			for (int w = 0; w < oW; w++)
			{
				int ih = h / Block;
				int iw = w / Block;

				int id = (h%Block)*Block + (w%Block);

				op[h*oW + w] = ip[ih*iW*iD + iw*iD + id];
			}
		}

		return _output;
	}

	bool DepthToSpace::operator()(std::vector<Tensor*> inputs)
	{
		if (inputs.size() != 1)
		{
			throw std::runtime_error("Expect 1 inputs!");
		}

		depth2space(inputs[0]);

		return true;
	}
}