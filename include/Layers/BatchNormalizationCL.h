// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\BatchNormalization.h"
#include "Layers\LayerCL.h"

namespace tf2ocl {

	class clBatchNorm;

	class BatchNormalizationCL : public BatchNormalization, public LayerCL {

	public:
		BatchNormalizationCL(const rapidjson::Value &config);

		virtual ~BatchNormalizationCL();

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

		clBatchNorm* _operator;
	};
}