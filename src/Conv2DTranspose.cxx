// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Conv2DTranspose.h"

#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include <iostream>

namespace tf2ocl {

	Conv2DTranspose::Conv2DTranspose(const rapidjson::Value &config) : Layer(config)
	{
		this->_kernel = NULL;
		this->_bias = NULL;

		_padding = config["padding"].GetString();
		if (_padding.compare("valid") == 0)
		{
			throw new std::runtime_error("Only same padding supported !");
		}

		const rapidjson::Value & strides = config["strides"];
		if (strides[0].GetInt() != 2 || strides[1].GetInt() != 2)
		{
			throw new std::runtime_error("Strides == 2 only supported !");
		}
		_strides[0] = strides[0].GetInt();
		_strides[1] = strides[1].GetInt();

		const rapidjson::Value & dilation = config["dilation_rate"];
		if (dilation[0].GetInt() != 1 || dilation[1].GetInt() != 1)
		{
			throw new std::runtime_error("Unsupported dilation rate for layer");
		}

		_activation = config["activation"].GetString();
		
		if (_activation.compare("relu") != 0 && _activation.compare("sigmoid") != 0 && _activation.compare("linear") != 0)
		{
			throw new std::runtime_error("Unsupported activation rate for layer");
		}

		const rapidjson::Value &weights = config["weights"];

		const rapidjson::Value &kernel = weights["kernel"];
		init_kernel(kernel);

		bool use_bias = config["use_bias"].GetBool();

		if (use_bias)
		{
			const rapidjson::Value &bias = weights["bias"];
			init_bias(bias);
		}
		//Check for strides and dilation!
	}

	Conv2DTranspose::~Conv2DTranspose()
	{
		if (this->_kernel)
			delete this->_kernel;

		if (this->_bias)
			delete this->_bias;
	}

	void Conv2DTranspose::init_bias(const rapidjson::Value& bias)
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

	void Conv2DTranspose::init_kernel(const rapidjson::Value& kernel)
	{
		if (this->_kernel == NULL)
		{
			//H W C D for transpose convolution
			const int H = kernel.Size();
			const int W = kernel[0].Size();
			const int C = kernel[0][0].Size();
			const int D = kernel[0][0][0].Size();

			if (H != 2 || W != 2)
			{
				throw new std::runtime_error("Kernel of size 2 only supported!");
			}

			TensorNDim kdim; kdim.push_back(H); kdim.push_back(W); kdim.push_back(D); kdim.push_back(C);
			this->_kernel = new Tensor(kdim, TensorDType::Float);
			this->_kernel->allocateHost();

			float* kp = (float*) this->_kernel->host();
			size_t index = 0;

			for (int h = 0; h < H; h++)//Column major for Eigen3
			{
				const rapidjson::Value& WKD = kernel[h];

				for (int w = 0; w < W; w++)
				{
					const rapidjson::Value& KD = WKD[w];

					for (int c = 0; c < C; c++)
					{
						const rapidjson::Value& DD = KD[c];

						for (int d = 0; d < D; d++)
						{
							//For conv2d transpose, kernel needs to be flipped just like convolution!
							int fh = 1 - h;
							int fw = 1 - w;
							index = fh*W*D*C + fw*D*C + d*C + c;
							kp[index] = DD[d].GetFloat();
						}
					}
				}
			}
		}
	}

	bool Conv2DTranspose::virtual_patch(Tensor *input, float* buffer, unsigned int i, unsigned j, int kH, int kW)
	{
		unsigned int iH = input->ndim()[0];
		unsigned int iW = input->ndim()[1];
		unsigned int D = input->ndim()[2];
		
		const float *in = (float*) input->host();
		
		size_t size = (kH) * (kW) * D;
		size_t index = 0;

		for (int _h = 0 ; _h < 2; _h++)
		{
			int oh = _h + i;
						
			for (int _w = 0; _w < 2; _w++)
			{
				int ow = _w + j;

				for (int d = 0; d < D; d++)
				{
					if (oh % 2 == 1 && ow % 2 == 1)
					{
						int h = (oh - 1) / 2;
						int w = (ow - 1) / 2;
						buffer[index++] = in[h*iW*D + w*D + d];
					}
					else
					{
						buffer[index++] = 0;
					}
				}
			}
		}

		return true;
	}

	bool Conv2DTranspose::operator()(std::vector<Tensor*> inputs)
	{
		Tensor* input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		conv2D(input,this->_kernel,this->_bias, this->_output);

		return true;
	}

	bool Conv2DTranspose::conv2D(Tensor* input, Tensor* kernel, Tensor* bias, Tensor* output)
	{
		unsigned int oH = output->ndim()[0];
		unsigned int oW = output->ndim()[1];
		unsigned int D = input->ndim()[2];
		unsigned int C = kernel->ndim()[3];

		unsigned int kH = kernel->ndim()[0];
		unsigned int kW = kernel->ndim()[1];

		unsigned int HWDSize = kH*kW*D;

		const float *ip = (float*)input->host();
		float *kp = (float*)kernel->host();
		Eigen::Map<Eigen::MatrixXf> ekl(kp, C, HWDSize);

		float *bp = NULL;

		if (bias)
			bp = (float*)bias->host();

		float *op = (float *)output->host();

		const bool useReLU = (_activation.compare("relu") == 0);
		const bool useSigmoid = (_activation.compare("sigmoid") == 0);

		float* buffer = new float[oW*HWDSize]{ 0 };

		//#pragma omp parallel for
		for (int h = 0; h < oH; h++)
		{
#pragma omp parallel for
			for (int w = 0; w < oW; w++)
			{
				float* patch = buffer + w*HWDSize;

				Conv2DTranspose::virtual_patch(input, patch, h, w, kH, kW);
			}

			Eigen::Map<Eigen::MatrixXf> ein(buffer, HWDSize, oW);

			Eigen::MatrixXf eo = ekl * ein;

			const float * eop = eo.data();

#pragma omp parallel for
			for (int w = 0; w < oW; w++)
			{
				unsigned int offset = h*(oW*C) + w*C;

				for (size_t l = 0; l < C; l++)
				{
					float v = eop[w*C + l];

					if (bp)
						v += bp[l];

					if (useReLU)
					{
						op[offset + l] = (v > 0) ? v : 0;
					}
					else if (useSigmoid)
					{
						op[offset + l] = 1.0 / (1.0 + std::exp(-v));
					}
					else
					{
						op[offset + l] = v;
					}
				}
			}
		}

		delete[] buffer;

		return true;
	}
}