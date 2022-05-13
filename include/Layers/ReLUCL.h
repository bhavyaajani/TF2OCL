// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\ReLU.h"
#include "Layers\LayerCL.h"

namespace tf2ocl {

	class clReLU;

	class ReLUCL : public ReLU, public LayerCL {

	public:
		ReLUCL(const rapidjson::Value &config);

		virtual ~ReLUCL();

		virtual bool compile();

		virtual bool operator()(std::vector<Tensor*> inputs);

	protected:
		virtual TensorMemType layerInputTensorMemType()
		{
			return TensorMemType::DEVICE;
		}

		virtual TensorMemType layerOutputTensorMemType()
		{
			return TensorMemType::DEVICE;
		}
		clReLU* _operator;
	};
}