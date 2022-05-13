// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\MaxPooling2D.h"
#include "Layers\LayerCL.h"

namespace tf2ocl {

	class clMaxPool2D;

	class MaxPooling2DCL : public MaxPooling2D, public LayerCL {

	public:
		MaxPooling2DCL(const rapidjson::Value &config);

		virtual ~MaxPooling2DCL();

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

		clMaxPool2D* _operator;
	};
}