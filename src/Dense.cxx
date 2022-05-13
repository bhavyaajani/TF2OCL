// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Dense.h"

#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include <iostream>

namespace tf2ocl {

	Dense::Dense(const rapidjson::Value &config) : Layer(config)
	{
		this->_kernel = NULL;
		this->_bias = NULL;

		_activation = config["activation"].GetString();
		if (_activation.compare("relu") != 0 && _activation.compare("sigmoid") != 0 && _activation.compare("softmax") != 0 && _activation.compare("linear") != 0)
		{
			throw new std::runtime_error("Unsupported activation rate for layer");
		}

		const rapidjson::Value &weights = config["weights"];

		const rapidjson::Value &kernel = weights["kernel"];
		init_kernel(kernel);

		const rapidjson::Value &bias = weights["bias"];
		init_bias(bias);
	}

	Dense::~Dense()
	{
		if (this->_kernel)
			delete this->_kernel;

		if (this->_bias)
			delete this->_bias;
	}

	void Dense::init_bias(const rapidjson::Value& bias)
	{
		if (this->_bias == NULL)
		{
			const int L = bias.Size();

			TensorNDim bdim; bdim.push_back(L);
			this->_bias = new Tensor(bdim, TensorDType::Float);
			this->_bias->allocateHost();

			float* bp = (float*) this->_bias->host();

			for (int l = 0; l < L; l++)
			{
				bp[l] = bias[l].GetFloat();
			}
		}
	}

	void Dense::init_kernel(const rapidjson::Value& kernel)
	{
		if (this->_kernel == NULL)
		{
			const int W = kernel.Size();
			const int H = kernel[0].Size();

			TensorNDim kdim; kdim.push_back(H); kdim.push_back(W);
			this->_kernel = new Tensor(kdim, TensorDType::Float);
			this->_kernel->allocateHost();

			float* kp = (float*) this->_kernel->host();
			size_t index = 0;

			for (int c = 0; c < W; c++)//Column major for Eigen3
			{
				const rapidjson::Value& col = kernel[c];

				for (int r = 0; r < H; r++)
				{
					kp[index++] = col[r].GetFloat();
				}
			}
		}
	}

	bool Dense::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		float *ip = (float*) input->host();
		Eigen::Map<Eigen::MatrixXf> ein(ip, input->ndim()[0], 1);

		float *kp = (float*) this->_kernel->host();
		Eigen::Map<Eigen::MatrixXf> ekl(kp, this->_kernel->ndim()[0], this->_kernel->ndim()[1]);

		float *bp = (float*)this->_bias->host();
		Eigen::Map<Eigen::MatrixXf> ebi(bp, this->_bias->ndim()[0], 1);

		Eigen::MatrixXf eo = (ekl * ein) + ebi;

		float *op = (float *) _output->host();
		const float * eop = eo.data();

		if (_activation.compare("relu") == 0)
		{
			for (size_t l = 0; l < _output->ndim()[0]; l++)
			{
				const float v = eop[l];
				op[l] = (v > 0) ? v : 0;
			}
		}
		else if (_activation.compare("softmax") == 0)
		{
			double norm = 1e-8;

			for (size_t l = 0; l < _output->ndim()[0]; l++)
			{
				double v = std::exp(eop[l]);
				op[l] = float(v);
				norm += v;
			}

			for (unsigned int l = 0; l < _output->ndim()[0]; l++)
			{
				op[l] = (float) (op[l] / norm);
			}
		}
		else if (_activation.compare("sigmoid") == 0)
		{
			for (unsigned int l = 0; l < _output->ndim()[0]; l++)
			{
				const float v = 1.0 / (1.0 + std::exp(-eop[l]));
				op[l] = v;
			}
		}
		else
		{
			for (size_t l = 0; l < _output->ndim()[0]; l++)
			{
				const float v = eop[l];
				op[l] = v;
			}
		}

		return true;
	}
}