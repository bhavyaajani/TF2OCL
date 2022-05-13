// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "rapidjson/document.h"

#include "Layers\LayerFactory.h"
#include "Graph.h"

namespace tf2ocl {

	class Model {

	public:
		Model();
		virtual ~Model();

		bool load(std::string modelPath);

		Tensor* operator()(Tensor &input);

	protected:
		bool parse(std::string modelPath, rapidjson::Document &doc);

		void release_layers();

		void release_graph();

		Graph* _graph;

		LayerContainerType _layers;
	};
}