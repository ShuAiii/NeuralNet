//
//  main.cpp
//  Neural Network
//
//  Created by Kevin Zhang on 2019-12-14.
//  Copyright © 2019 Kevin Zhang. All rights reserved.
//

#include "Headers.hpp"

int main() {

	double learning_rate = 0.01;
	int epoch = 100;
	double lastError = 10000000;
	string dataAddress = "C:/Users/Jkzhang/source/repos/NeuralNet/weight_height.csv";
	int targetDim = 1;
	/*
	cout<<"Enter the Number of Prediction Targets:";
	cin>>targetDim;
	vector<int> posTarget;
	int pos;
	for(int i=0;i<targetDim;i++){
		cout<<"Enter the Column Number of Target "<<i+1<<":";
		cin>>pos;
		cout<<endl;
		posTarget.push_back(pos);
	}*/


	vector<vector<string>> classData;
	classData = readCsv(dataAddress);

	// Parsing This Dataset //
	int dimData;
	dimData = int(classData.size());
	vector<vector<double>> data;
	vector<double> aData;
	int sex;
	for (int i = 1; i < dimData; i++) {
		sex = classData[i][0].compare("\"Male\"");
		if (sex != 0) {
			aData.push_back(1);
		}
		else {
			aData.push_back(0);
		}
		for (size_t j = 1; j < classData[i].size(); ++j) {
			if (j == 1) {
				aData.push_back(atof(classData[i][j].c_str()) - 66);
			}
			else {
				aData.push_back(atof(classData[i][j].c_str()) - 135);
			}
		}
		data.push_back(aData);
		aData.clear();
	}
	// Parsing Data Done //

	// Spilting the Data into Training and Testing //
	data = shuffleData(data);
	int spiltsize;
	spiltsize = int(data.size()*0.8);
	vector<vector<double>> train(data.begin(), data.begin() + spiltsize);
	vector<vector<double>> test(data.begin() + spiltsize, data.end());
	// Finished Spilting //

	vector<vector<double>> pred;
	vector<vector<vector<double>>> grad;

	double error = 0;
	vector<vector<vector<double>>> iweight;
	vector<vector<double>> ibias;
	int numLayer;
	int numHidden;
	int dataLayer;

	cout << "Enter the Number of Hidden Layers:";
	cin >> numLayer;
	cout << endl;
	dataLayer = int(train[0].size()) - targetDim;
	vector<int> numNeuron;
	numNeuron.push_back(dataLayer);
	for (int i = 0; i < numLayer; i++) {
		cout << "Enter the Number of Neurons in Hidden Layer " << i + 1 << ":";
		cin >> numHidden;
		cout << endl;
		numNeuron.push_back(numHidden);
	}
	numNeuron.push_back(targetDim);
	iweight = initializeWeight(numNeuron, iweight);
	ibias = initializeBias(numNeuron, ibias);
	Net testnet(iweight, ibias);
	cout << "Net Constructed" << endl;
	for (int i = 0; i < epoch; i++) {
		data = shuffleData(train);
		for (size_t j = 0; j < data.size(); j++) {
			vector<double> y(train[j].begin(), train[j].begin() + targetDim);
			vector<double> x(train[j].begin() + targetDim, train[j].end());
			pred = testnet.feedforward(x);
			error = error + error_function(pred.back(), y);
			grad = testnet.backpropagation(pred, y);
			testnet.gradient_descent(grad, learning_rate);
		}

		if (error > lastError) {
			learning_rate = learning_rate * 0.9;
		}
		else {
			learning_rate = learning_rate * 1.1;
		}

		cout << "-----Error of Batch " << i << " -----:" << error << endl;
		lastError = error;
		error = 0;
	}

	// Validation //
	int correct = 0;
	ofstream myFile("C:/Users/Jkzhang/source/repos/NeuralNet/Prediction.csv");
	for (size_t j = 0; j < test.size(); j++) {
		vector<double> y(test[j].begin(), test[j].begin() + targetDim);
		vector<double> x(test[j].begin() + targetDim, test[j].end());
		pred = testnet.feedforward(x);
		if (pred.back()[0] - y[0] < 0.5) {
			correct = correct + 1;
		}
		myFile << pred.back()[0];
		for (size_t i = 0; i < test[j].size(); i++) {
			myFile << ",";
			myFile << test[j][i];
		}
		myFile << "\n";
	}
	cout << "Validation Accuracy %" << double(correct) / test.size() * 100 << endl;
	myFile.close();
	// Validation Finished //
	return 0;
}