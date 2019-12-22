//
//  Neuron.cpp
//  Neural Network
//
//  Created by Kevin Zhang on 2019-12-14.
//  Copyright © 2019 Kevin Zhang. All rights reserved.
//
#include "Headers.hpp"

Neuron::Neuron(std::vector<double> &weight, double &bias) {
	m_weight = weight;
	m_bias = bias;
};

Neuron::~Neuron() {
};

double Neuron::feedforward(std::vector<double> &input) {
	double lin_comb = 0;
	double *plin_comb = &lin_comb;
	*plin_comb = m_weight * input + m_bias;
	return sigmoid(*plin_comb);
};



void Neuron::gradient_descent(std::vector<double> &gradient, double learning_rate) {
	for (size_t i = 0; i < gradient.size() - 1; i++) {
		m_weight[i] = m_weight[i] - learning_rate * gradient[i];
	}
	m_bias = m_bias - learning_rate * gradient.back();
};

std::vector<double> Neuron::backpropagation(std::vector<double> &prediction) {
	std::vector<double> gradient;
	double lin_comb = 0;
	double *plin_comb = &lin_comb;
	*plin_comb = m_weight * prediction + m_bias;
	for (size_t i = 0; i < m_weight.size(); i++) {
		gradient.push_back(prediction[i] * sigmoid_derivative(*plin_comb));
	}
	gradient.push_back(sigmoid_derivative(*plin_comb));
	return gradient;
};

std::vector<double> Neuron::backpropagation2(std::vector<double> &prediction) {
	std::vector<double> gradient;
	double lin_comb = 0;
	double *plin_comb = &lin_comb;
	*plin_comb = m_weight * prediction + m_bias;
	for (size_t i = 0; i < m_weight.size(); i++) {
		gradient.push_back(m_weight[i] * sigmoid_derivative(*plin_comb));
	}
	return gradient;
};

std::vector<double> Neuron::backpropagation3(std::vector<double> &prediction, std::vector<double> &trueValue) {
	std::vector<double> gradient;
	for (size_t i = 0; i < prediction.size(); i++) {
		gradient.push_back(-2 * (trueValue[i] - prediction[i]));
	}
	return gradient;
};