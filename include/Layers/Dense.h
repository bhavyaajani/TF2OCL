// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Layer.h"

namespace tf2ocl {

	class Dense : public Layer {

	public:
		Dense(const rapidjson::Value &config);

		virtual ~Dense();

		virtual bool operator()(std::vector<Tensor*> inputs);

	protected:
		void init_kernel(const rapidjson::Value& kernel);
		void init_bias(const rapidjson::Value& bias);

		std::string _activation;

		Tensor* _kernel;
		Tensor* _bias;
	};
}