// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Model.h"

#include "rapidjson/document.h"
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <iostream>

#include "Layers\LayerFactory.h"

#include <chrono>

namespace tf2ocl {

	Model::Model() {
		_graph = NULL;
	}

	Model::~Model() {
		
		release_layers();
		release_graph();
	}

	void Model::release_graph()
	{
		if (_graph)
			delete _graph;
		_graph = NULL;
	}

	void Model::release_layers()
	{
		LayerContainerType::iterator itr;

		for (itr = _layers.begin(); itr != _layers.end(); ++itr)
		{
			delete (itr->second);
		}

		this->_layers.clear();
	}

	bool Model::parse(std::string modelPath, rapidjson::Document &doc)
	{
		FILE* fp = fopen(modelPath.c_str(), "r");
		fseek(fp, 0, SEEK_END);
		size_t filesize = (size_t)ftell(fp);
		fseek(fp, 0, SEEK_SET);

		char* buffer = (char*)malloc(filesize + 1);
		size_t readLength = fread(buffer, 1, filesize, fp);
		buffer[readLength] = '\0';
		fclose(fp);

		doc.ParseInsitu(buffer);

		return true;
	}

	bool Model::load(std::string modelPath)
	{
		release_layers();
		release_graph();

		auto m_start = std::chrono::high_resolution_clock::now();
		rapidjson::Document model_config;
		bool status = parse(modelPath, model_config);

		auto m_stop = std::chrono::high_resolution_clock::now();
		auto m_time = std::chrono::duration_cast<std::chrono::milliseconds>(m_stop - m_start);
		std::cout << "Model Load " << " : " << m_time.count() << " ms ---------->" << std::endl;

		if (!status)
		{
			return false;
		}

		if (!model_config.HasMember("layers"))
		{
			return false;
		}

		const rapidjson::Value& layer_config = model_config["layers"];

		status = LayerFactory::build_layers(layer_config, _layers);

		if (!status)
		{
			release_layers();
			return false;
		}

		_graph = new Graph();
		status = _graph->compile(model_config);

		if (!status) {
			release_layers();
			release_graph();
		}

		return status;
	}

	Tensor* Model::operator()(Tensor &input)
	{
		if (!_graph)
			return NULL;

		
		LayerContainerType::iterator itr;

		for (itr = _layers.begin(); itr != _layers.end(); ++itr)
		{
			itr->second->setDirty();
		}

		Tensor* op = _graph->call(input, _layers);

		if (op)
		{
			//Get final output in host!
			if (op->defaultMemType() == TensorMemType::DEVICE)
			{
				op->allocateHost();
				op->memcopyDeviceToHost();
			}
		}

		return op;
	}
}