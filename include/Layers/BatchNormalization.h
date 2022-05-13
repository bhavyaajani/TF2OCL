// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Layer.h"

namespace tf2ocl {

	class BatchNormalization : public Layer {

	public:
		BatchNormalization(const rapidjson::Value &config);

		virtual ~BatchNormalization();

		virtual bool operator()(std::vector<Tensor*> inputs);

	protected:
		Tensor* batchNormalization(Tensor *input);

		void init_gamma(const rapidjson::Value &config);
		void init_beta(const rapidjson::Value &config);

		void init_mean(const rapidjson::Value &config);
		void init_variance(const rapidjson::Value &config, float epsilon);

		Tensor* _gamma;
		Tensor* _beta;
		Tensor* _mean;
		Tensor* _variance;

		float _epsilon;
	};
}