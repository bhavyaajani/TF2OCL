// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Layer.h"

namespace tf2ocl {

	class Concatenate : public Layer {

	public:
		Concatenate(const rapidjson::Value &config);

		virtual ~Concatenate();

		virtual bool operator()(std::vector<Tensor*> inputs);

	protected:
		Tensor* concate(Tensor *first, Tensor *second);
	};
}