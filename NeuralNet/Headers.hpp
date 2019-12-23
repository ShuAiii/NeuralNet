//
//  Headers.hpp
//  Neural Network
//
//  Created by Kevin Zhang on 2019-12-14.
//  Copyright © 2019 Kevin Zhang. All rights reserved.
//
#ifndef Headers_hpp

#define Headers_hpp

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

//// Neural Net Obejects  ////
class Neuron {
	friend class Layer;
private:
	vector<double> m_weight;
	double m_bias;
public:
	Neuron(const vector<double> &weight, const double &bias);
	~Neuron();
	double feedforward(const vector<double> &input);
	vector<double> backpropagation(const vector<double> &prediction);
	vector<double> backpropagation2(const vector<double> &prediction);
	vector<double> backpropagation3(const vector<double> &prediction, const vector<double> &trueValue);
	void gradient_descent(const vector<double> &gradient, double learning_rate);

};

class Layer {
	friend class Net;
private:
	vector<Neuron*> layer;
	vector<vector<double> > m_weight;
	vector<double> m_bias;
public:
	Layer(const vector<vector<double> > &weight, const vector<double> &bias);
	~Layer();
	vector<double> feedforward(const vector<double> &input);
	vector<vector<double>> backpropagation(const vector<double> &prediction);
	vector<vector<double>> backpropagation2(const vector<double> &prediction);
	vector<vector<double>> backpropagation3(const vector<double> &prediction, const vector<double> &trueValue);
	void gradient_descent(const vector<vector<double>> &gradient, double learning_rate);
};



class Net {
private:
	vector<Layer*> net;
	vector<vector<vector<double>>> m_weight;
	vector<vector<double>> m_bias;
public:
	Net(const vector<vector<vector<double> >> &weight, const vector<vector<double> > &bias);
	~Net();
	vector<vector<double>> feedforward(const vector<double> &input);
	vector<vector<vector<double>>> backpropagation(const vector<vector<double>> &prediction, const vector<double> &trueValue);
	void gradient_descent(const vector<vector<vector<double>>> &gradient, double learning_rate);
};

//// Helper Functions and Objects ////

double error_function(vector<double> prediction, vector<double> trueValue);

double sigmoid(double x);
double sigmoid_derivative(double x);

double dot(const vector<double> &vec1, const vector<double> &vec2);
double operator*(const vector<double> &vec1, const vector<double> &vec2);

vector<vector<double>> multiply(vector<vector<double>> &mat1, vector<vector<double>> &mat2);
vector<vector<double>> operator*(vector<vector<double>> mat1, vector<vector<double>> mat2);

vector<double> multiply(vector<vector<double>> &mat, vector<double> &vec);

vector<double> operator*(vector<vector<double>> mat, vector<double> vec);

vector<vector<double>> cross_multiply(vector<double> &vec, vector<vector<double>> &mat);

vector<double> collapse(vector<vector<double>> &mat);

vector<vector<vector<double>>> initializeWeight(const vector<int> &numNeuron, vector<vector<vector<double>>> &iweight);

vector<vector<double>> initializeBias(const vector<int> & numNeuron, vector<vector<double>> &ibias);

double norm_rand();

vector<vector<string>> readCsv(const string &address);

vector<vector<double>> shuffleData(vector<vector<double>> &data);
#endif /* Headers_hpp */
