#include "stdafx.h"
#include "Resource.h"

int main()
{
	vector<Image> Train;
	vector<Image> Test;
	vector<vector<double>> wG;
	vector<vector<double>> bG;
	vector<double> Output;
	int n = 2;
	int Epochs = 40;
	int batchsize = 100;
	float learningRate = 10;

	readMnist(Train, n, "lib/train-images.idx3-ubyte", "lib/train-labels.idx1-ubyte");
	readMnist(Test, 0, "lib/test-images.idx3-ubyte", "lib/test-labels.idx1-ubyte");
	Net Topology({ 784, 16, 16, 10 }, true);
	clock_t start, end;
	start = clock();

	//Topology.Train(Train, Test, Epochs, n, batchsize, learningRate, true, false);

	end = clock();
	cout << "Execution time : " << float(end - start) / CLOCKS_PER_SEC << " Seconds" << '\n';

	return 0;
}