// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Graph.h"

#include <chrono>

namespace tf2ocl {

	Graph::Graph() {
		_graph = NULL;
	}

	Graph::~Graph() {
		release_graph();
	}

	void Graph::release_graph()
	{
		if (_graph)
			delete _graph;
		_graph = NULL;
	}

	bool Graph::compile(rapidjson::Document & model_config)
	{
		release_graph();

		if (!model_config.HasMember("output"))
		{
			return false;
		}

		const rapidjson::Value& layer_config = model_config["layers"];

		std::string output = model_config["output"].GetString();
		_graph = new Node(output);	

		bool status = build_graph(_graph, layer_config);

		if (!status)
		{
			release_graph();
		}

		return status;
	}

	bool Graph::build_graph(Node* parent, const rapidjson::Value& layer_config)
	{
		for (rapidjson::SizeType i = 0; i < layer_config.Size(); i++)
		{
			const rapidjson::Value& layer = layer_config[i];
			const std::string name = layer["name"].GetString();

			if (name.compare(parent->_name) == 0)
			{
				const rapidjson::Value& childrens = layer["input_layers"];
				
				bool status = true;//If no child, its a valid terminal node!

				for (int c = 0; c < childrens.Size(); c++)
				{
					Node * child = new Node(childrens[c].GetString());
					status &= build_graph(child, layer_config);
					if (status)
					{
						parent->_inputs.push_back(child);
					}
					else
					{
						delete child;
						return false;
					}
				}

				return status;
			}
		}

		return false;//If parent layer does not exist!
	}

	Tensor* Graph::call(Tensor &input, std::map<std::string, Layer*> &layers)
	{
		if (!_graph)
			return NULL;

		std::vector<Tensor*> inputs;
		inputs.push_back(&input);

		//auto m_start = std::chrono::high_resolution_clock::now();
		Tensor* op = _call(_graph, inputs, layers);
		//auto m_stop = std::chrono::high_resolution_clock::now();
		//auto m_time = std::chrono::duration_cast<std::chrono::milliseconds>(m_stop - m_start);
		//std::cout << "Graph: " << m_time.count() << " ms ---------->" << std::endl;

		return op;
	}

	Tensor* Graph::_call(Node* node, std::vector<Tensor*> inputs, std::map<std::string, Layer*> &layers)
	{
		bool terminal = (node->_inputs.size() == 0);

		if (terminal)
		{
			std::map<std::string, Layer*>::iterator itr = layers.find(node->_name);
			if (itr != layers.end())
			{
				Tensor* op = itr->second->call(inputs);
				return op;
			}
			else
			{
				return NULL;
			}
		}

		std::vector<Tensor*> node_inputs;

		for (int n = 0; n < node->_inputs.size(); n++)
		{
			std::map<std::string, Layer*>::iterator itr = layers.find(node->_inputs[n]->_name);
			if (itr != layers.end())
			{
				Tensor* op = NULL;

				if(itr->second->isDirty())
					op = this->_call(node->_inputs[n], inputs, layers);
				else
					op = itr->second->call(inputs);

				if (op)
					node_inputs.push_back(op);
				else
					return NULL;
			}
			else
			{
				return NULL;
			}
		}

		
		std::map<std::string, Layer*>::iterator itr = layers.find(node->_name);
		if (itr != layers.end())
		{
			Tensor* op = itr->second->call(node_inputs);
			return op;
		}
		else
		{
			return NULL;
		}		
	}
}