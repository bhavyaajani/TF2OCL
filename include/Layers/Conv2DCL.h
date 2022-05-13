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

	class Conv2DCL : public Conv2D, public LayerCL {

	public:
		Conv2DCL(const rapidjson::Value &config);

		virtual ~Conv2DCL();

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

		clConv2D* _operator;
		Tensor* _kernelCHWD;
	};
}