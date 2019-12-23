//
//  Net.cpp
//  Neural Network
//
//  Created by Kevin Zhang on 2019-12-14.
//  Copyright © 2019 Kevin Zhang. All rights reserved.
//
#include "Headers.hpp"

Net::Net(const vector<vector<vector<double>>> &weight, const vector<vector<double> > &bias) {
	m_weight = weight;
	m_bias = bias;
	for (size_t i = 0; i < weight.size(); i++) {
		//Layer layer(m_weight[i],m_bias[i]);
		//net.push_back(layer);
		//std::shared_ptr<Layer> p_layer(new Layer(m_weight[i],m_bias[i]));
		//net.push_back(p_layer.get());
		Layer *p_layer = new Layer(m_weight[i], m_bias[i]);
		net.push_back(p_layer);
	}
};

Net::~Net() {
};

vector<vector<double>> Net::feedforward(const vector<double> &input) {
	vector<vector<double>> prediction;
	prediction.push_back(input);
	for (size_t i = 0; i < m_weight.size(); i++) {
		prediction.push_back(net[i]->feedforward(prediction[i]));
	};
	return prediction;
};

vector<vector<vector<double>>> Net::backpropagation(const vector<vector<double>> &prediction, const vector<double> &trueValue) {
	vector<vector<vector<double>>> gradient;
	vector<vector<double>> jacobi_mat;
	vector<double> jacobi_sum;
	for (size_t i = 0; i < m_weight.size(); i++) {
		gradient.push_back(net[i]->backpropagation(prediction[i]));
	}
	jacobi_mat = net.back()->backpropagation3(prediction.back(), trueValue);
	jacobi_sum = collapse(jacobi_mat);
	gradient.back() = cross_multiply(jacobi_sum, gradient.back());
	for (size_t i = m_weight.size() - 1; i > 0; i--) {
		jacobi_mat = jacobi_mat * (net[i]->backpropagation2(prediction[i]));
		jacobi_sum = collapse(jacobi_mat);
		gradient[i - 1] = cross_multiply(jacobi_sum, gradient[i - 1]);
	};
	return gradient;
};

void Net::gradient_descent(const vector<vector<vector<double>>> &gradient, double learning_rate) {
	for (size_t i = 0; i < gradient.size(); i++) {
		net[i]->gradient_descent(gradient[i], learning_rate);
	};
};