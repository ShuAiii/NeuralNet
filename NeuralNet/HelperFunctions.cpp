//
//  Helper_Funs.cpp
//  Neural Network
//
//  Created by Kevin Zhang on 2019-12-14.
//  Copyright © 2019 Kevin Zhang. All rights reserved.
//
#include "Headers.hpp"

double error_function(std::vector<double> prediction, std::vector<double> trueValue) {
	double error = 0;
	double *p_error = &error;
	for (size_t i = 0; i < prediction.size(); i++) {
		*p_error = *p_error + pow((trueValue[i] - prediction[i]), 2);
	}
	return *p_error;
};

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
};

double sigmoid_derivative(double x) {
	double fx;
	fx = sigmoid(x);
	return fx * (1 - fx);
};

double dot(std::vector<double> &vec1, std::vector<double> &vec2) {
	double answer = 0;
	double *p_answer = &answer;
	for (size_t i = 0; i < vec1.size(); i++) {
		*p_answer = *p_answer + vec1[i] * vec2[i];
	};
	return *p_answer;
};

double operator*(std::vector<double> &vec1, std::vector<double> &vec2) {
	return dot(vec1, vec2);
};

std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> &mat1, std::vector<std::vector<double>> &mat2) {
	int row = int(mat1.size());
	int col = int(mat2[0].size());
	int cross = int(mat2.size());
	std::vector<std::vector<double>> answer(row, std::vector<double>(col, 0));
	double dotsum;
	double *p_dotsum = &dotsum;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			*p_dotsum = 0;
			for (int k = 0; k < cross; k++) {
				*p_dotsum = *p_dotsum + mat1[i][k] * mat2[k][j];
			}
			answer[i][j] = *p_dotsum;
		}
	}
	return answer;
};

std::vector<std::vector<double>> operator*(std::vector<std::vector<double>> mat1, std::vector<std::vector<double>> mat2) {
	return multiply(mat1, mat2);
};

std::vector<double> multiply(std::vector<std::vector<double>> &mat, std::vector<double> &vec) {
	std::vector<double> answer(mat.size(), 0);
	double dotsum;
	double *p_dotsum = &dotsum;
	for (size_t i = 0; i < mat.size(); i++) {
		*p_dotsum = 0;
		for (size_t k = 0; k < vec.size(); k++) {
			*p_dotsum = *p_dotsum + mat[i][k] * vec[k];
		}
		answer[i] = *p_dotsum;
	}
	return answer;
};

std::vector<double> operator*(std::vector<std::vector<double>> mat, std::vector<double> vec) {
	return multiply(mat, vec);
};

std::vector<std::vector<double>> cross_multiply(std::vector<double> &vec, std::vector<std::vector<double>> &mat) {
	int row = int(mat.size());
	int col = int(mat[0].size());
	std::vector<std::vector<double>> answer(row, std::vector<double>(col, 0));
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			answer[i][j] = vec[i] * mat[i][j];
		}
	}
	return answer;
};

std::vector<double> collapse(std::vector<std::vector<double>> &mat) {
	int row = int(mat.size());
	int col = int(mat[0].size());
	std::vector<double> answer(col, 0);
	for (int j = 0; j < col; j++) {
		for (int i = 0; i < row; i++) {
			answer[j] = answer[j] + mat[i][j];
		}
	}
	return answer;
};

std::vector<std::vector<std::vector<double>>> initializeWeight(const std::vector<int> &numNeuron, std::vector<std::vector<std::vector<double>>> &iweight) {
	std::vector<double> iweightVec;
	std::vector<std::vector<double>> iweightMat;
	for (size_t i = 1; i < numNeuron.size(); i++) {
		for (int j = 0; j < numNeuron[i]; j++) {
			for (int k = 0; k < numNeuron[i - 1]; k++) {
				iweightVec.push_back(norm_rand());
			}
			iweightMat.push_back(iweightVec);
			iweightVec.clear();
		}
		iweight.push_back(iweightMat);
		iweightMat.clear();
	}
	return iweight;
};

std::vector<std::vector<double>> initializeBias(const std::vector<int> & numNeuron, std::vector<std::vector<double>> &ibias) {
	std::vector<double> ibiasVec;
	for (size_t i = 1; i < numNeuron.size(); i++) {
		for (int j = 0; j < numNeuron[i]; j++) {
			ibiasVec.push_back(norm_rand());
		}
		ibias.push_back(ibiasVec);
		ibiasVec.clear();
	}
	return ibias;
};

double norm_rand() {
	double normRV;
	double one = 1.0;
	normRV = rand()*one / (RAND_MAX*one);
	return normRV;
};

vector<vector<string>> readCsv(const string &address) {
	ifstream inFile(address);
	vector<vector<string>> classData;
	vector<string> samplePoint;
	string line;

	while (getline(inFile, line, '\n'))
	{
		stringstream ss(line);
		string dataPoint;
		while (std::getline(ss, dataPoint, ',')) {
			samplePoint.push_back(dataPoint);
		}
		classData.push_back(samplePoint);
		samplePoint.clear();
	}
	return classData;
};

vector<vector<double>> shuffleData(vector<vector<double>> &data) {
	vector<vector<double>> copyData;
	int ind;
	copyData = data;
	for (int i = 0; i < data.size(); i++) {
		ind = rand() % copyData.size();
		data[i] = copyData[ind];
		copyData.erase(copyData.begin() + ind);
	}
	return data;
};