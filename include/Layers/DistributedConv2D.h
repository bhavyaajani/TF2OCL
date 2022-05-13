// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Conv2D.h"
#include "Layers\LayerCL.h"

namespace tf2ocl {

	class clConv2D;

	class DistributedConv2D : public Conv2D, public LayerCL {

	public:
		DistributedConv2D(const rapidjson::Value &config);

		virtual ~DistributedConv2D();

		virtual bool compile();

		virtual bool operator()(std::vector<Tensor*> inputs);

	protected:
		virtual void build_outputs();

		virtual void init_kernel(const rapidjson::Value& kernel);
		virtual void init_bias(const rapidjson::Value& bias);

		clConv2D* _operator;
		Tensor* _kernel1;
		Tensor* _bias1;

		Tensor* _kernel2;
		Tensor* _bias2;

		Tensor* _output1;
		Tensor* _output2;

		int _ocL;
	};
}