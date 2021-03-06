//
//  Layer.cpp
//  Neural Network
//
//  Created by Kevin Zhang on 2019-12-14.
//  Copyright � 2019 Kevin Zhang. All rights reserved.
//
#include "Headers.hpp"

Layer::Layer(const vector<vector<double> > &weight, const vector<double> &bias) {
	m_weight = weight;
	m_bias = bias;
	for (size_t i = 0; i < weight.size(); i++) {
		//std::shared_ptr<Neuron> p_neuron(new Neuron(m_weight[i],m_bias[i]));
		//layer.push_back(p_neuron.get());
		Neuron *p_neuron = new Neuron(m_weight[i], m_bias[i]);
		layer.push_back(p_neuron);
	}
};

Layer::~Layer() {
};

vector<double> Layer::feedforward(const vector<double> &input) {
	vector<double> prediction;
	for (size_t i = 0; i < m_weight.size(); i++) {
		prediction.push_back(layer[i]->feedforward(input));
	};
	return prediction;
};

void Layer::gradient_descent(const vector<vector<double>> &gradient, double learning_rate) {
	for (size_t i = 0; i < gradient.size(); i++) {
		layer[i]->gradient_descent(gradient[i], learning_rate);
	};
};

vector<vector<double>> Layer::backpropagation(const vector<double> &prediction) {
	vector<vector<double>> gradient;
	for (size_t i = 0; i < m_weight.size(); i++) {
		gradient.push_back(layer[i]->backpropagation(prediction));
	};
	return gradient;
};

vector<vector<double>> Layer::backpropagation2(const vector<double> &prediction) {
	vector<vector<double>> gradient;
	for (size_t i = 0; i < m_weight.size(); i++) {
		gradient.push_back(layer[i]->backpropagation2(prediction));
	};
	return gradient;
};

vector<vector<double>> Layer::backpropagation3(const vector<double> &prediction, const vector<double> &trueValue) {
	vector<vector<double>> gradient;
	for (size_t i = 0; i < m_weight.size(); i++) {
		gradient.push_back(layer[i]->backpropagation3(prediction, trueValue));
	};
	return gradient;
};