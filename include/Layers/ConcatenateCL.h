// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Concatenate.h"
#include "Layers\LayerCL.h"

namespace tf2ocl {

	class clConcate;

	class ConcatenateCL : public Concatenate, public LayerCL {

	public:
		ConcatenateCL(const rapidjson::Value &config);

		virtual ~ConcatenateCL();

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
		clConcate* _operator;
	};
}