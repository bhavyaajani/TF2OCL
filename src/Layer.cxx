// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Layer.h"
#include <iostream>

namespace tf2ocl {

	Layer::Layer(const rapidjson::Value &config)
	{
		_dirty = true;
		_output = NULL;

		this->_name = config["name"].GetString();

		this->_type = config["type"].GetString();

		if (config.HasMember("data_format"))
		{
			std::string data_format = config["data_format"].GetString();
			if (data_format.compare("channels_last") != 0)
			{
				std::cout << "Invalid data format for layer!" << std::endl;
				throw new std::runtime_error("Invalid data format !");
			}
		}

		const rapidjson::Value& indim = config["input_shape"];
		for (rapidjson::SizeType s = 0; s < indim.Size(); s++)
		{
			this->_inputNDim.push_back(indim[s].GetInt());
		}

		const rapidjson::Value& odim = config["output_shape"];
		for (rapidjson::SizeType s = 0; s < odim.Size(); s++)
		{
			this->_outputNDim.push_back(odim[s].GetInt());
		}

	}

	Layer::~Layer()
	{
		if (_output)
			delete _output;
	}

	void Layer::build_outputs()
	{
		if (_output)
			return;
				
		_output = new Tensor(this->_outputNDim, TensorDType::Float);
		
		TensorMemType memType = layerOutputTensorMemType();
		switch (memType)
		{
		case tf2ocl::TensorMemType::HOST:
			_output->allocateHost();
			_output->setDefaultMemType(TensorMemType::HOST);
			break;
		case tf2ocl::TensorMemType::DEVICE:
			_output->allocateDevice(CL_MEM_READ_WRITE);//Because next op shall read from it !
			_output->setDefaultMemType(TensorMemType::DEVICE);
			break;
		default:
			break;
		}
	}

	void Layer::check_inputs(std::vector<Tensor*> inputs)
	{
		for (auto T : inputs)
		{
			if (T)
			{
				TensorMemType memType = layerInputTensorMemType();

				if (memType != T->defaultMemType())
				{
					switch (memType)
					{
					case tf2ocl::TensorMemType::HOST:
						T->allocateHost();
						T->memcopyDeviceToHost();
						break;
					case tf2ocl::TensorMemType::DEVICE:
						//All subsequent ops shall only do READ on input tensor!
						T->allocateDevice(CL_MEM_READ_ONLY);
						T->memcopyHostToDevice();
						break;
					default:
						break;
					}
				}
			}
			else
			{
				throw new std::runtime_error("NULL input not supported !");
			}
		}
	}

	Tensor* Layer::call(std::vector<Tensor*> inputs)
	{
		this->build_outputs();

		if (_dirty)
		{
			check_inputs(inputs);
			this->operator()(inputs);
			_dirty = false;
		}

		return this->_output;
	}
}