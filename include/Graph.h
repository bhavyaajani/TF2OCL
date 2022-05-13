// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include <vector>
#include <map>

#include <iostream>

#include "rapidjson/document.h"
#include "Layers\Layer.h"

namespace tf2ocl {

	class Node {

	public :
		Node(std::string name) 
		{
			_name = name;
		}

		~Node()
		{
			for (Node* node : _inputs)
			{
				delete node;
			}
			_inputs.clear();
		}

	std::string _name;
	std::vector<Node* > _inputs;
	};

	class Graph {

	public:
		Graph();
		virtual ~Graph();

		bool compile(rapidjson::Document & model_config);

		Tensor* call(Tensor &input, std::map<std::string, Layer*> &layers);

	protected:
		Tensor* _call(Node* node, std::vector<Tensor*> inputs, std::map<std::string, Layer*> &layers);

		bool build_graph(Node* parent, const rapidjson::Value& layer_config);

		void release_graph();

		Node * _graph;
	};
}