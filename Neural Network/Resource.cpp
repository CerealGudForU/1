#include "stdafx.h"
#include "Resource.h"

weight::weight(int inNum, double assign)
{
	this->val = assign;
}

weight::weight(int inNum)
{
	this->val = rand_range(-1 / sqrt(inNum), 1 / sqrt(inNum));
}

Neuron::Neuron(int totalIn, vector<double> wAssign, double bAssign)
{
	for (int i = 0; i < totalIn; i++) {
		this->weights.emplace_back(totalIn, wAssign[i]);
	}
	this->bias = bAssign;
}

Neuron::Neuron(int totalIn)
{
	for (int i = 0; i < totalIn; i++) {
		this->weights.emplace_back(totalIn);
	}
	this->bias = 0;
}

Net::Net(vector<int> structure, bool useData)
{
	if (useData == true) {
		fstream fWeight;
		fstream fBias;
		vector<vector<vector<double>>> wAssign;
		vector<vector<double>> bAssign;
		fWeight.open("lib/weights.txt", ios::in);
		if (fWeight.fail()) {
			cout << '\n' << "Failed to open weights.txt" << '\n';
		}
		else {
			for (int i = 1; i < structure.size(); i++) {
				wAssign.emplace_back();
				for (int j = 0; j < structure[i]; j++) {
					wAssign[i - 1].emplace_back();
					for (int k = 0; k < structure[i - 1]; k++) {
						double temp;
						fWeight >> temp;
						wAssign[i - 1][j].push_back(temp);
					}
				}
			}
			fWeight.close();
		}
		fBias.open("lib/biases.txt", ios::in);
		if (fBias.fail()) {
			cout << '\n' << "Failed to open biases.txt" << '\n';
		}
		else {
			for (int i = 1; i < structure.size(); i++) {
				bAssign.emplace_back();
				for (int j = 0; j < structure[i]; j++) {
					double temp;
					fBias >> temp;
					bAssign[i - 1].emplace_back();
				}
			}
			fBias.close();
		}
		for (int i = 1; i < structure.size(); i++) {
			p_layers.emplace_back();
			for (int j = 0; j < structure[i]; j++) {
				p_layers[i - 1].emplace_back(structure[i - 1], wAssign[i - 1][j], bAssign[i - 1][j]);
			}
		}
	}
	else {
		for (int i = 1; i < structure.size(); i++) {
			p_layers.emplace_back();
			for (int j = 0; j < structure[i]; j++) {
				p_layers[i - 1].emplace_back(structure[i - 1]);
			}
		}
	}
}

void Net::Compute(const vector<double>& Input, vector<double>& Outputs, int& FinalOutput, float& cost, int Desired = NULL) const
{
	vector<vector<double>> Inputs;
	Inputs.push_back(Input);
	for (int i = 0; i < p_layers.size(); i++) {
		vector<double> temp;
		for (int j = 0; j < p_layers[i].size(); j++) {
			temp.emplace_back();
			for (int k = 0; k < Inputs[i].size(); k++) {
				temp[j] += p_layers[i][j].weights[k].val * Inputs[i][k];
			}
			temp[j] += p_layers[i][j].bias;
			//cout << p_layers[i][j].weights[0].val << " " << temp[j] << '\n';
			Sigmoid(temp[j]);
		}
		Inputs.push_back(temp);
	}
	Outputs = Inputs[Inputs.size() - 1];
	bool temp = true;
	for (int i = 0; i < Outputs.size(); i++) {
		int x{};
		for (int j = 0; j < Outputs.size(); j++) {
			if (Outputs[i] > Outputs[j]) {
				x++;
			}
		}
		if (temp == true && x == 9) {
			FinalOutput = i;
			temp = false;
		}
		else if (temp == true) {
			FinalOutput = -1;
		}
	}
	if (Desired != NULL) {
		cost = 0;
		for (int i = 0; i < Outputs.size(); i++) {
			cost += ((i == Desired) ? (Outputs[i] - 1) * (Outputs[i] - 1) : (Outputs[i] + 1) * (Outputs[i] + 1));
		}
	}
}

void Net::GradientDescent(const vector<double>& Input, const int Desired, vector<double>& wOutput, vector<double>& bOutput)
{
	vector<vector<double>> Inputs;
	vector<vector<double>> z;
	Inputs.push_back(Input);
	for (int i = 0; i < p_layers.size(); i++) {
		vector<double> temp;
		for (int j = 0; j < p_layers[i].size(); j++) {
			temp.emplace_back();
			for (int k = 0; k < Inputs[i].size(); k++) {
				temp[j] += p_layers[i][j].weights[k].val * Inputs[i][k];
			}
			temp[j] += p_layers[i][j].bias;
		}
		z.push_back(temp);
		for (int j = 0; j < temp.size(); j++) {
			Sigmoid(temp[j]);
		}
		Inputs.push_back(temp);
	}
	vector<vector<double>> ac;
	for (int i = 0; i < p_layers.size(); i++) {
		ac.emplace_back();
	}
	for (int i = 0; i < p_layers[p_layers.size() - 1].size(); i++) {
		ac[ac.size() - 1].push_back((i == Desired) ? 2 * (Inputs[Inputs.size() - 1][i] - 1) : 2 * (Inputs[Inputs.size() - 1][i] + 1));
	}
	for (int i = p_layers.size() - 2; i > -1; i--) {
		for (int j = 0; j < p_layers[i].size(); j++) {
			double temp{};
			for (int k = 0; k < p_layers[i + 1].size(); k++) {
				temp += p_layers[i + 1][k].weights[j].val * dSigmoid(z[i + 1][k]) * ac[i + 1][k];
			}
			ac[i].push_back(temp);
		}
	}
	wOutput.clear();
	bOutput.clear();
	for (int i = 0; i < p_layers.size(); i++) {
		for (int j = 0; j < p_layers[i].size(); j++) {
			for (int k = 0; k < p_layers[i][j].weights.size(); k++) {
				wOutput.push_back(Inputs[i][k] * dSigmoid(z[i][j]) * ac[i][j]);
			}
			bOutput.push_back(dSigmoid(z[i][j]) * ac[i][j]);
		}
	}
}

void Net::Apply(const vector<double>& wInput, const vector<double>& bInput, const float& learningRate)
{
	int x{};
	int y{};
	for (int i = 0; i < p_layers.size(); i++) {
		for (int j = 0; j < p_layers[i].size(); j++) {
			for (int k = 0; k < p_layers[i][j].weights.size(); k++) {
				double temp = this->p_layers[i][j].weights[k].val;
				this->p_layers[i][j].weights[k].val -= (learningRate * wInput[x]);
				//cout << wInput[x] << '\n';
				x++;
			}
			this->p_layers[i][j].bias -= learningRate * bInput[y];
			y++;
		}
	}
}

void Net::Train(vector<Image> Input, const vector<Image>& Test, const int Epochs, const int imgNum, const int batchsize, const float learningRate, const bool Compare, const bool storeData) {
	vector<vector<double>> wG;
	vector<vector<double>> bG;
	int epochCounter{};
	for (int i = 0; i < imgNum; i++) {
		wG.emplace_back();
		bG.emplace_back();
	}
	if (imgNum % batchsize == 0) {
		for (int i = 0; i < Epochs; i++) {
			int counter{};
			int index{};
			if (imgNum != batchsize) {
				random_shuffle(Input.begin(), Input.end());
			}
			for (int j = 0; j < imgNum / batchsize; j++) {
				for (int l = 0; l < batchsize; l++) {
					GradientDescent(Input[index].normPixels, Input[index].label, wG[index], bG[index]);
					index++;
				}
				mean(wG, batchsize);
				mean(bG, batchsize);
				this->Apply(wG[0], bG[0], learningRate);
				counter++;
			}
			epochCounter++;
			cout << "Epoch: " << epochCounter << " / " << Epochs << " Complete" << '\n';
			if (storeData == true) {
				fstream wFile;
				fstream bFile;
				fstream EpochNum;
				wFile.open("lib/weights.txt", ios::out);
				if (wFile.fail()) {
					cout << '\n' << "Failed to write weights.txt" << '\n';
				}
				else {
					for (int i = 0; i < p_layers.size(); i++) {
						for (int j = 0; j < p_layers[i].size(); j++) {
							for (int k = 0; k < p_layers[i][j].weights.size(); k++) {
								wFile << p_layers[i][j].weights[k].val << '\n';
							}
						}
					}
					wFile.close();
				}
				bFile.open("lib/biases.txt", ios::out);
				if (bFile.fail()) {
					cout << '\n' << "Failed to write weights.txt" << '\n';
				}
				else {
					for (int i = 0; i < p_layers.size(); i++) {
						for (int j = 0; j < p_layers[i].size(); j++) {
							bFile << p_layers[i][j].bias << '\n';
						}
					}
					bFile.close();
				}
				EpochNum.open("lib/Epoch Num.txt", ios::out);
				if (EpochNum.fail()) {
					cout << '\n' << "Failed to write to Epoch Num.txt" << '\n';
				}
				else {
					//Zero based counting
					EpochNum << i << '\n';
					EpochNum.close();
				}
			}
			if (Compare == true && i % 4 == 0) {
				vector<double> Output;
				float CostS{};
				int FinalOut;
				float Cost;
				float rw{};
				for (int j = 0; j < imgNum; j++) {
					this->Compute(Input[j].normPixels, Output, FinalOut, Cost, Input[j].label);
					CostS += Cost;
					(FinalOut == Input[j].label) ? rw += 1 : rw += 0;
				}
				cout << '\n' << "Epoch: " << i + 1 << '\n';
				cout << "Dataset Cost: " << CostS / imgNum << '\n';
				cout << "Dataset Accuracy: " << rw / imgNum * 100 << '%' << '\n' << '\n';
				CostS = 0;
				rw = 0;
				for (int j = 0; j < Test.size(); j++) {
					this->Compute(Test[j].normPixels, Output, FinalOut, Cost, Test[j].label);
					CostS += Cost;
					(FinalOut == Test[j].label) ? rw += 1 : rw += 0;
				}
				cout << "Testing Cost: " << CostS / Test.size() << '\n';
				cout << "Testing Accuracy: " << rw / Test.size() * 100 << '%' << '\n' << '\n';
			}
		}
	}
	else {
		cout << '\n' << "Bathsize is not proportional to imgNum, " << '\n';
	}
}

void readMnist(vector<Image>&InOut, int n, string imgPath, string labelPath)
{
	fstream img;
	fstream label;
	img.open(imgPath, ios::binary | ios::in);
	if (!img.fail()) {
		img.seekg(16, ios::beg);
		for (int i = 0; i < n; i++) {
			InOut.emplace_back();
			for (int j = 0; j < 784; j++) {
				uchar buffer = 0;
				img.read((char*)&buffer, sizeof(buffer));
				InOut[i].pixels.push_back(buffer);
				InOut[i].normPixels.push_back(((double)buffer) / 255);
			}
		}
		img.close();
	}
	else {
		cout << '\n' << "Error opening image file" << '\n';
	}
	label.open(labelPath, ios::binary | ios::in);
	if (!label.fail()) {
		label.seekg(8, ios::beg);
		for (int i = 0; i < n; i++) {
			uchar buffer;
			label.read((char*)&buffer, sizeof(uchar));
			InOut[i].label = (int)buffer;
		}
		label.close();
	}
	else {
		cout << '\n' << "Error opening label file" << '\n';
	}

}

double rand_range(double min, double max) {
	return min + (max - min) * ((double)rand()) / RAND_MAX;
}

void Sigmoid(double& x) {
	x = tanh(x);
}

double dSigmoid(double x) {
	return 1 - tanh(x) * tanh(x);
}

void mean(vector<vector<double>>& x, int Num)
{
	vector<double> temp;
	for (int i = 0; i < x[0].size(); i++) {
		double buffer{};
		for (int j = 0; j < Num; j++) {
			buffer += x[j][i];
		}
		buffer /= x.size();
		temp.push_back(buffer);
	}
	x[0] = temp;
}
