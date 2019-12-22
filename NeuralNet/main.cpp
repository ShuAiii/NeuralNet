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
	double lastError = 10000000;
	string dataAddress = "C:/Users/Jkzhang/Desktop/C++/weight_height.csv";
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
	dataLayer = int(data[0].size()) - targetDim;
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

	vector<double> x;
	vector<double> y;
	int n = 10;
	for (int i = 0; i < n; i++) {
		data = shuffleData(data);
		for (size_t j = 0; j < data.size(); j++) {
			vector<double> y(data[j].begin(), data[j].begin() + targetDim);
			vector<double> x(data[j].begin() + targetDim, data[j].end());
			pred = testnet.feedforward(x);
			error = error + error_function(pred.back(), y);
			grad = testnet.backpropagation(pred, y);
			testnet.gradient_descent(grad, learning_rate);
		}
		/*
		if (error > lastError){
			learning_rate = learning_rate*0.9;
		}
		else {
			learning_rate = learning_rate*1.1;
		}
		*/
		cout << "-----Error of Batch " << i << " -----:" << error << endl;
		lastError = error;
		error = 0;
	}
	//cout << "-----True Value-----:"<<trueValue[0]<<endl;


	// Validation //
	vector<double> test1;
	vector<double> test2;
	vector<double> test3;
	vector<double> test4;

	test1.push_back(-3);
	test1.push_back(-40);
	test2.push_back(6);
	test2.push_back(65);
	test3.push_back(8);
	test3.push_back(55);
	test4.push_back(5);
	test4.push_back(8);

	vector<double> result1;
	vector<double> result2;
	vector<double> result3;
	vector<double> result4;

	result1 = testnet.feedforward(test1).back();
	result2 = testnet.feedforward(test2).back();
	result3 = testnet.feedforward(test3).back();
	result4 = testnet.feedforward(test4).back();

	cout << "-----Prediction Angela-----:" << result1[0] << endl;
	cout << "-----Prediction John-----:" << result2[0] << endl;
	cout << "-----Prediction Kevin-----:" << result3[0] << endl;
	cout << "-----Prediction Alina-----:" << result4[0] << endl;

	ofstream myFile("C:/Users/Jkzhang/Desktop/C++/Prediction.csv");

	// Send data to the stream
	myFile << result1[0];
	myFile << "\n";
	myFile << result2[0];
	myFile << "\n";
	myFile << result3[0];
	myFile << "\n";
	myFile << result4[0];
	myFile << "\n";

	// Close the file
	myFile.close();

	return 0;
}