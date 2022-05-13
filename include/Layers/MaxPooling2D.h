// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Layer.h"

namespace tf2ocl {

	class MaxPooling2D : public Layer {

	public:
		MaxPooling2D(const rapidjson::Value &config);

		virtual ~MaxPooling2D();

		virtual bool operator()(std::vector<Tensor*> inputs);

	protected:
		void max2D(Tensor *input, float *out, unsigned int i, unsigned j, int ph, int pw);

		Tensor* maxPool2D(Tensor *input, int ph, int pw, int sh, int sw);

		TensorNDim _pool_size;
		TensorNDim _strides;
	};
}