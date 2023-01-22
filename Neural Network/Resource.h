#pragma once
#include "stdafx.h"

struct Image {
	vector<double> normPixels;
	vector<unsigned char> pixels;
	int label;
};

struct weight {
	double val;
	weight();
	weight(int inNum, double assign);
	weight(int inNum);
};

struct Neuron {
	vector<weight> weights;
	double bias;
	Neuron();
	Neuron(int totalIn, vector<double> wAssign, double bAssign);
	Neuron(int totalIn);
};

struct Net {
	vector<vector<Neuron>> p_layers;
	Net(vector<int> structure, bool useData);
	void Compute(const vector<double>& Input, vector<double>& Outputs, int& FinalOutput, float& cost, int Desired) const;
	void GradientDescent(const vector<double>& Input, const int Desired, vector<double>& wOutput, vector<double>& bOutput);
	void Apply(const vector<double>& wInput, const vector<double>& bInput, const float& learningRate);
	void Train(vector<Image> Input, const vector<Image>& Test, const int Epochs, const int imgNum, const int batchsize, const float learningRate, const bool Compare, const bool storeData);
};

void readMnist(vector<Image>& InOut, int n, string imgPath, string labelPath);
double rand_range(double min, double max);
void Sigmoid(double& x);
double dSigmoid(double x);
void mean(vector<vector<double>>& x, int Num);
