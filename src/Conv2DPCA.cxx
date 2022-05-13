// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Conv2DPCA.h"

#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>
#include <iostream>

namespace tf2ocl {

	Conv2DPCA::Conv2DPCA(const rapidjson::Value &config) : Layer(config)
	{
		this->_kernel = NULL;
		this->_bias = NULL;
		this->_kernel_pca = NULL;

		_padding = config["padding"].GetString();

		const rapidjson::Value & strides = config["strides"];
		if (strides[0].GetInt() != 1 || strides[1].GetInt() != 1)
		{
			if (_padding.compare("same") == 0)
			{
				throw new std::runtime_error("Strides > 1 supported with 'Valid' padding only!");
			}
		}
		_strides[0] = strides[0].GetInt();
		_strides[1] = strides[1].GetInt();

		const rapidjson::Value & dilation = config["dilation_rate"];
		if (dilation[0].GetInt() != 1 || dilation[1].GetInt() != 1)
		{
			throw new std::runtime_error("Unsupported dilation rate for layer");
		}

		_activation = config["activation"].GetString();

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

	Conv2DPCA::~Conv2DPCA()
	{
		if (this->_kernel)
			delete this->_kernel;

		if (this->_bias)
			delete this->_bias;
	}

	void Conv2DPCA::init_bias(const rapidjson::Value& bias)
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

	void Conv2DPCA::init_kernel(const rapidjson::Value& kernel)
	{
		if (this->_kernel == NULL)
		{
			const int H = kernel.Size();
			const int W = kernel[0].Size();
			const int D = kernel[0][0].Size();
			const int C = kernel[0][0][0].Size();

			if (H % 2 == 0 || W % 2 == 0)
			{
				throw new std::runtime_error("Even kernel size not supported!");
			}

			TensorNDim kdim; kdim.push_back(H); kdim.push_back(W); kdim.push_back(D); kdim.push_back(C);
			this->_kernel = new Tensor(kdim, TensorDType::Float);
			this->_kernel->allocateHost();

			float* kp = (float*) this->_kernel->host();
			size_t index = 0;

			for (int h = 0; h < H; h++)//Column major for Eigen3
			{
				const rapidjson::Value& WDK = kernel[h];

				for (int w = 0; w < W; w++)
				{
					const rapidjson::Value& DK = WDK[w];

					for (int d = 0; d < D; d++)
					{
						const rapidjson::Value& K = DK[d];

						for (int c = 0; c < C; c++)
						{
							float value = K[c].GetFloat();
							kp[index++] = value;
						}
					}
				}
			}

			init_pca_kernel();
		}
	}

	void Conv2DPCA::init_pca_kernel()
	{
		unsigned int kH = _kernel->ndim()[0];
		unsigned int kW = _kernel->ndim()[1];
		unsigned int D = _kernel->ndim()[2];
		unsigned int C = _kernel->ndim()[3];

		unsigned int count = kH*kW*D*C;

		float* kp = new float[count];
		memcpy(kp, _kernel->host(), sizeof(float)*count);

		unsigned int HWDSize = kH*kW*D;
		Eigen::Map<Eigen::MatrixXf> KMat(kp, C, HWDSize);

		Eigen::MatrixXf Cov = KMat * KMat.transpose() / (C - 1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(Cov);
		
		const float* peval = eig.eigenvalues().data();

		float eig_sum = 0;

		for (int i = 0; i < C; i++)
		{
			eig_sum += std::abs(peval[i]);//Close to 0 values may be negative!
		}

		if (eig_sum < 1e-6)
		{
			std::cout << "Eigen values are too small" << std::endl;
		}

		float eigen_cumulative = 0;
		
		int K = 0;

		for (int i = C-1; i >= 0; i--)
		{
			eigen_cumulative += std::abs(peval[i]) / eig_sum;
			K++;

			if (eigen_cumulative > 0.95)
			{
				break;
			}
		}

		//std::cout << "Using " << K << " out of " << C << std::endl;

		//K = C;

		const float* evp = eig.eigenvectors().data();// C X C
		float* vp = new float[C*K]; // C x K

		int index = 0;
		for (int i = (C - K)*C; i < C*C; i++)
		{
			vp[index++] = evp[i];
		}

		Eigen::Map<Eigen::MatrixXf> V(vp, C, K);

		TensorNDim evdim; evdim.push_back(C); evdim.push_back(K);
		this->_eigen_vec = new Tensor(evdim, TensorDType::Float);
		this->_eigen_vec->allocateHost();
		
		memcpy(_eigen_vec->host(), V.data(), sizeof(float)*C*K);

		Eigen::MatrixXf U = V.transpose() * KMat; // K x HWD

		TensorNDim kdim; kdim.push_back(kH); kdim.push_back(kW); kdim.push_back(D); kdim.push_back(K);
		this->_kernel_pca = new Tensor(kdim, TensorDType::Float);
		this->_kernel_pca->allocateHost();

		memcpy(_kernel_pca->host(), U.data(), sizeof(float)*kH*kW*D*K);

		delete[] kp;
		delete[] vp;

		//Eigen::VectorXf KHat = V * U;
		//std::cout << "Check PCA: " << (K - KHat).norm() << std::endl; //Should be close to 0
	}

	bool Conv2DPCA::virtual_patch(Tensor *input, float* buffer, unsigned int i, unsigned j, int kH, int kW)
	{
		unsigned int iH = input->ndim()[0];
		unsigned int iW = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		//Expect kH and kW to be odd !
		int offH = kH / 2;
		int offW = kW / 2;
		
		const float *in = (float*) input->host();
		
		size_t size = (kH) * (kW) * D;
		size_t index = 0;

		for (int _h = -offH ; _h <= offH; _h++)
		{
			int h = _h + i;
			bool invalid = (h < 0 || h >= iH);
			if (invalid) {
				memset(buffer + index, 0, sizeof(float)* D*kW);
				index += D*kW;
				continue;
			}

			for (int _w = -offW; _w <= offW; _w++)
			{
				int w = _w + j;
				invalid = (w < 0 || w >= iW);

				if (invalid) {
					memset(buffer + index, 0, sizeof(float)*D);
					index += D;
					continue;
				}

				for (int d = 0; d < D; d++)
				{
					buffer[index++] = in[h*iW*D + w*D + d];
				}
			}
		}

		return true;
	}

	bool Conv2DPCA::operator()(std::vector<Tensor*> inputs)
	{
		Tensor* input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		conv2D(input,this->_kernel_pca,this->_bias);

		return true;
	}

	Tensor* Conv2DPCA::conv2D(Tensor* input, Tensor* kernel, Tensor* bias)
	{
		unsigned int oH = _output->ndim()[0];
		unsigned int oW = _output->ndim()[1];
		unsigned int D = input->ndim()[2];
		unsigned int K = kernel->ndim()[3];
		unsigned int _C = _eigen_vec->ndim()[0];

		unsigned int kH = kernel->ndim()[0];
		unsigned int kW = kernel->ndim()[1];

		unsigned int HWDSize = kH*kW*D;

		const float *ip = (float*)input->host();
		float *kp = (float*)kernel->host();
		Eigen::Map<Eigen::MatrixXf> ekl(kp, K, HWDSize);

		float *evp = (float*)_eigen_vec->host();
		Eigen::Map<Eigen::MatrixXf> evpm(evp, _C, K);

		float *bp = NULL;

		if (bias)
			bp = (float*)bias->host();
		
		float *op = (float *)_output->host();

		int offH = (_padding.compare("same") == 0) ? 0 : kH / 2;
		int offW = (_padding.compare("same") == 0) ? 0 : kW / 2;

		const bool useReLU = (_activation.compare("relu") == 0);
		const bool useSigmoid = (_activation.compare("sigmoid") == 0);

		float* buffer = new float[oW*HWDSize]{ 0 };

//#pragma omp parallel for
		for (int h = 0; h < oH; h++)
		{
			int ih = h*_strides[0] + offH;

#pragma omp parallel for
			for (int w = 0; w < oW; w++)
			{
				int iw = w*_strides[1] + offW;

				float* patch = buffer + w*HWDSize;

				Conv2DPCA::virtual_patch(input, patch ,ih, iw, kH, kW);
			}

			Eigen::Map<Eigen::MatrixXf> ein(buffer, HWDSize, oW);

			Eigen::MatrixXf eo = evpm * (ekl * ein);

			const float * eop = eo.data();

#pragma omp parallel for
			for (int w = 0; w < oW; w++)
			{
				unsigned int offset = h*(oW*_C) + w*_C;

				for (size_t l = 0; l < _C; l++)
				{
					float v = eop[w*_C + l];
					
					if(bp)
						v+=bp[l];

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

		return _output;
	}
}