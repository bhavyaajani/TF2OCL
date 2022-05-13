// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\DistributedConv2D.h"
#include "OpenCL\clConv2D.h"

#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include <iostream>
#include <chrono>

namespace tf2ocl {

	DistributedConv2D::DistributedConv2D(const rapidjson::Value &config) : Conv2D(config), LayerCL()
	{		
		_operator = new clConv2D();

		this->_kernel1 = NULL;
		this->_kernel2 = NULL;
		this->_bias1 = NULL;
		this->_bias2 = NULL;

		unsigned int applySigmoid = (_activation.compare("sigmoid") == 0);
		if (applySigmoid)
		{
			throw std::runtime_error("DistributedConv2D does not support Sigmoid activation yet!");
		}

		const rapidjson::Value &weights = config["weights"];

		const rapidjson::Value &kernel = weights["kernel"];
		const int C = kernel[0][0][0].Size();
		this->_ocL = int(0.5*C);

		init_kernel(kernel);

		bool use_bias = config["use_bias"].GetBool();

		if (use_bias)
		{
			const rapidjson::Value &bias = weights["bias"];
			init_bias(bias);
		}
	}

	DistributedConv2D::~DistributedConv2D()
	{
		if (this->_kernel1)
			delete this->_kernel1;

		if (this->_bias1)
			delete this->_bias1;

		if (this->_kernel2)
			delete this->_kernel2;

		if (this->_bias2)
			delete this->_bias2;

		if (_operator)
			delete _operator;
	}

	void DistributedConv2D::init_bias(const rapidjson::Value& bias)
	{
		if (this->_bias1 == NULL)
		{
			const int L = bias.Size();
			
			TensorNDim bdim1; bdim1.push_back(_ocL);
			this->_bias1 = new Tensor(bdim1, TensorDType::Float);
			this->_bias1->allocateHost();

			float* bp1 = (float*) this->_bias1->host();

			for (int l = 0; l < _ocL; l++)
			{
				bp1[l] = bias[l].GetFloat();
			}

			TensorNDim bdim2; bdim2.push_back(L - _ocL);
			this->_bias2 = new Tensor(bdim2, TensorDType::Float);
			this->_bias2->allocateHost();

			float* bp2 = (float*) this->_bias2->host();

			for (int l = _ocL; l < L; l++)
			{
				bp2[l-(L/2)] = bias[l].GetFloat();
			}
		}
	}

	void DistributedConv2D::init_kernel(const rapidjson::Value& kernel)
	{
		if (this->_kernel1 == NULL)
		{
			const int H = kernel.Size();
			const int W = kernel[0].Size();
			const int D = kernel[0][0].Size();
			const int C = kernel[0][0][0].Size();

			if (H % 2 == 0 || W % 2 == 0)
			{
				throw new std::runtime_error("Even kernel size not supported!");
			}

			TensorNDim kdim1; kdim1.push_back(H); kdim1.push_back(W); kdim1.push_back(D); kdim1.push_back(_ocL);
			this->_kernel1 = new Tensor(kdim1, TensorDType::Float);
			this->_kernel1->allocateHost();

			float* kp1 = (float*) this->_kernel1->host();

			TensorNDim kdim2; kdim2.push_back(H); kdim2.push_back(W); kdim2.push_back(D); kdim2.push_back(C-_ocL);

			this->_kernel2 = new Tensor(kdim2, TensorDType::Float);
			this->_kernel2->allocateHost();

			float* kp2 = (float*) this->_kernel2->host();

			size_t index1 = 0;
			size_t index2 = 0;

			for (int h = 0; h < H; h++)//Column major for Eigen3
			{
				const rapidjson::Value& WDK = kernel[h];

				for (int w = 0; w < W; w++)
				{
					const rapidjson::Value& DK = WDK[w];

					for (int d = 0; d < D; d++)
					{
						const rapidjson::Value& K = DK[d];

						for (int c = 0; c < _ocL; c++)
						{
							kp1[index1++] = K[c].GetFloat();
						}

						for (int c = _ocL; c < C; c++)
						{
							kp2[index2++] = K[c].GetFloat();
						}
					}
				}
			}
		}
	}

	bool DistributedConv2D::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}

		return _compiled;
	}

	void DistributedConv2D::build_outputs()
	{
		if (_output)
			return;

		TensorNDim odim = this->_outputNDim;

		TensorNDim dim1; dim1.push_back(odim[0]); dim1.push_back(odim[1]); dim1.push_back(_ocL);
		this->_output1 = new Tensor(dim1, TensorDType::Float);
		this->_output1->allocateHost();

		TensorNDim dim2; dim2.push_back(odim[0]); dim2.push_back(odim[1]); dim2.push_back(odim[2]-_ocL);
		this->_output2 = new Tensor(dim2, TensorDType::Float);
		this->_output2->allocateHost();

		_output = new Tensor(odim, TensorDType::Float);
		_output->allocateHost();
	}

	bool DistributedConv2D::operator()(std::vector<Tensor*> inputs)
	{
		Tensor* input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		float *X = (float*)input->host();
		float *K1 = (float*) this->_kernel1->host();
		//float *K = (float*) this->_kernelCHWD->host();
		float *B1 = NULL;
		if (this->_bias1)
			B1 = (float*)this->_bias1->host();

		unsigned int iH = input->ndim()[0];
		unsigned int iW = input->ndim()[1];
		unsigned int D = input->ndim()[2];
		
		unsigned int oH = this->_output->ndim()[0];
		unsigned int oW = this->_output->ndim()[1];
		unsigned int  C = this->_output->ndim()[2];

		unsigned int kH = this->_kernel1->ndim()[0];
		unsigned int kW = this->_kernel1->ndim()[1];

		unsigned int sH = _strides[0];
		unsigned int sW = _strides[1];

		unsigned int applyReLU = (_activation.compare("relu") == 0);
		unsigned int applyPadding = (_padding.compare("same") == 0);

		unsigned int C1 = _output1->ndim()[2];
		unsigned int C2 = _output2->ndim()[2];

		_operator->call(input->device(), _kernel1->device(), _bias1->device(), _output1->device(),
			iH, iW, D, kH, kW, oH, oW, C1, sH, sW, applyPadding, applyReLU);
		//_operator->call(X, K1, B1, iH, iW, D, kH, kW, oH, oW, C1, sH, sW, applyPadding, applyReLU);

		conv2D(input, this->_kernel2, this->_bias2, _output2);
		
		float* op1 = (float*)_output1->host();

		unsigned int size1 = (oH*oW*C1)*sizeof(float);
				
		float* op2 = (float*)_output2->host();

		float* op = (float*)_output->host();

#pragma omp parallel for
		for (int h = 0; h < oH; h++)
		{
			for (int w = 0; w < oW; w++)
			{
				for (int c = 0; c < _ocL; c++)
				{
					op[h*oW*C + w*C + c] = op1[h*oW*C1 + w*C1 + c];
				}
				for (int c = _ocL; c < C; c++)
				{
					op[h*oW*C + w*C + c] = op2[h*oW*C2 + w*C2 + c - _ocL];
				}
			}
		}
		
		return true;
	}	
}