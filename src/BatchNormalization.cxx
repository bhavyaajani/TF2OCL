// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\BatchNormalization.h"
#include <iostream>

namespace tf2ocl {

	BatchNormalization::BatchNormalization(const rapidjson::Value &config) : Layer(config)
	{
		this->_gamma = NULL;
		this->_beta = NULL;

		this->_mean = NULL;
		this->_variance = NULL;

		const rapidjson::Value &weights = config["weights"];

		bool useGamma = config["scale"].GetBool();
		if (useGamma)
		{
			const rapidjson::Value &gamma = weights["gamma"];
			init_gamma(gamma);
		}

		bool useBeta = config["center"].GetBool();
		if (useBeta)
		{
			const rapidjson::Value &beta = weights["beta"];
			init_beta(beta);
		}

		const rapidjson::Value &mean = weights["mean"];
		init_mean(mean);

		this->_epsilon = config["epsilon"].GetFloat();

		const rapidjson::Value &variance = weights["variance"];
		init_variance(variance, _epsilon);

	}

	BatchNormalization::~BatchNormalization()
	{
	}

	void BatchNormalization::init_gamma(const rapidjson::Value &gamma)
	{
		if (this->_gamma == NULL)
		{
			const int L = gamma.Size();

			TensorNDim bdim; bdim.push_back(L);
			this->_gamma = new Tensor(bdim, TensorDType::Float);
			this->_gamma->allocateHost();

			float* bp = (float*) this->_gamma->host();

			for (int l = 0; l < L; l++)
			{
				bp[l] = gamma[l].GetFloat();
			}
		}
	}

	void BatchNormalization::init_beta(const rapidjson::Value &beta)
	{
		if (this->_beta == NULL)
		{
			const int L = beta.Size();

			TensorNDim bdim; bdim.push_back(L);
			this->_beta = new Tensor(bdim, TensorDType::Float);
			this->_beta->allocateHost();

			float* bp = (float*) this->_beta->host();

			for (int l = 0; l < L; l++)
			{
				bp[l] = beta[l].GetFloat();
			}
		}
	}

	void BatchNormalization::init_mean(const rapidjson::Value &mean)
	{
		if (this->_mean == NULL)
		{
			const int L = mean.Size();

			TensorNDim bdim; bdim.push_back(L);
			this->_mean = new Tensor(bdim, TensorDType::Float);
			this->_mean->allocateHost();

			float* bp = (float*) this->_mean->host();

			for (int l = 0; l < L; l++)
			{
				bp[l] = mean[l].GetFloat();
			}
		}
	}

	void BatchNormalization::init_variance(const rapidjson::Value &var, float epsilon)
	{
		if (this->_variance == NULL)
		{
			const int L = var.Size();

			TensorNDim bdim; bdim.push_back(L);
			this->_variance = new Tensor(bdim, TensorDType::Float);
			this->_variance->allocateHost();

			float* bp = (float*) this->_variance->host();

			for (int l = 0; l < L; l++)
			{
				float variance = var[l].GetFloat();
				bp[l] = std::sqrt(variance + this->_epsilon);
			}
		}
	}

	Tensor* BatchNormalization::batchNormalization(Tensor *input)
	{
		unsigned int H = input->ndim()[0];
		unsigned int W = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		const unsigned int count = W*H;

		float *ip = (float *)input->host();
		float *op = (float *)_output->host();

		float *gp = (this->_gamma) ? (float *)_gamma->host() : NULL;
		float *bp = (this->_beta) ? (float *)_beta->host() : NULL;

		float *mp = (float *)_mean->host();
		float *vp = (float *)_variance->host();

#pragma omp parallel for
		for (int d = 0; d < D; d++)
		{
			float gamma = (gp) ? gp[d] : 1;
			float beta =  (bp) ? bp[d] : 0;

			float mean = mp[d];
			float variance = vp[d];

			for (int i = 0; i < count; i++)
			{
				unsigned int offset = i*D + d;

				op[offset] = gamma * ( (ip[offset] - mean)/ variance) + beta;
			}
		}

		return _output;
	}

	bool BatchNormalization::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		//This is same as class _output!
		batchNormalization(input);
		return true;
	}
}