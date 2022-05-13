// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Layer.h"

namespace tf2ocl {

	class Conv2D : public Layer {

	public:
		Conv2D(const rapidjson::Value &config);

		virtual ~Conv2D();

		virtual bool operator()(std::vector<Tensor*> inputs);

	protected:
		bool virtual_patch(Tensor *input, float* buffer, unsigned int i, unsigned j, int kH, int kW);
		bool conv2D(Tensor* input, Tensor* kernel, Tensor* bias, Tensor* output);

		virtual void init_kernel(const rapidjson::Value& kernel);
		virtual void init_bias(const rapidjson::Value& bias);

		std::string _activation;

		Tensor* _kernel;
		Tensor* _bias;
		int _strides[2];
		std::string _padding;
	};
}