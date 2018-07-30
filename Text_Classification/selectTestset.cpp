/*
  We select the training set and test set
  Input: paper_labels.txt	
  Training set: train.txt	
  Test set: test.txt	
  Input arguments: the percentage of the test set
*/
#include<fstream>
#include<iostream>
#include<ctime>
#include<string>
#include<map>
#include<set>
#include<iterator>
#include<assert.h>
using namespace std;

#define DATANUM 19396
map<int, int> dataset;
set<int> testSet;
set<int> trainSet;

// Randomly select (%index)data as test data and the remain are training data
void getRandom(int index)
{
	testSet.clear();
	trainSet.clear();
	int testNum = DATANUM*index / 100.0;
	for (int i = 0; i < testNum; ++i)
	{
		while (1)
		{
			int id = (rand() % 19396) + 1;
			if (testSet.find(id) == testSet.end())
				if (dataset[id] != 0)
				{
					testSet.insert(id);
					break;
				}
		}
	}
	for (int i = 1; i <= DATANUM; ++i)
	{
		if (testSet.find(i) == testSet.end())
			trainSet.insert(i);
	}
	cout << "train_set\\testset selected!\n";
}

void getdata(string file)
{
	ifstream in = ifstream(file);
	assert(in.is_open());
	int paper, label;
	while (in >> paper >> label)
	{
		dataset.insert(make_pair(paper, label));
	}
	in.close();
}

int main()
{
	string infile = "paper_labels.txt";
	int index;
	cout << "test num(%index): ";
	cin >> index;
	getdata(infile);
	char num[] = {'0','1','2','3','4','5','6','7','8','9'};
	for(int i = 0;i<10;++i)
	{
		string trainSetFile = "./Sets/train";
		string testSetFile = "./Sets/test";
		string Filetype = ".txt";
		trainSetFile +=num[i] + Filetype;
		testSetFile += num[i] + Filetype;
		getRandom(index);
		ofstream out = ofstream(trainSetFile);
		set<int>::iterator pos = trainSet.begin();
		while (pos != trainSet.end())
		{
			out << *pos << endl;
			pos++;
		}
		out.close();
		out = ofstream(testSetFile);
		pos = testSet.begin();
		while (pos != testSet.end())
		{
			out << *pos << endl;
			pos++;
		}
		out.close();
	}
}
