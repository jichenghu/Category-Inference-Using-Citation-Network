/*
	1.Naivebayes doesn't train or test the samples without labels
	when compiling：
		1.using TEST：get the accuracy rate when the test set occupies %20 to %40 of the data set 
			output file: "NaiveBayes_score.txt"
			format：the_proportion_of_the_test_set + "\t" + accuracy_rate
		2.using GETPROB：get the most likely label and the probability of the label
			output file: "paper_real_label_prob.txt"
			format：paper + "\t" + true_label + "\t" + calcu_label + "\t" + prob
*/

#include<fstream>
#include<iostream>
#include<cstring>
#include<string>
#include<set>
#include<map>
#include<ctime>
#include<cmath>
#include<iterator>
#include<assert.h>
using namespace std;

#define DATANUM 19396

set<int> testSet;
set<int> trainSet;

struct Datasets
{
	//  key: paper  val: label
	map<int, int> papers_and_labels;

	//  key: paper  val: terms
	map<int, set<int> >  papers_and_terms;

	//  key: term  val: paper
	map<int, set<int> > terms_and_papers;

	map<int, int> labelSet;

	//dictionary
	set<int> termSet;

	void openfile(string fileName, ifstream& readin)
	{
		readin = ifstream(fileName);
		assert(readin.is_open());
	}

	void closefile(ifstream& readin)
	{
		readin.close();
	}

	//get data(papers, terms, labels)
	Datasets(string PL_filePath,string PT_filePath)
	{
		ifstream readin;
		int id1, id2;

		cout << "get Dataset..." << endl;

		openfile(PL_filePath, readin);
		while (readin >> id1 >> id2)
		{
			papers_and_labels.insert(make_pair(id1, id2));
			if (id2 != 0)
			{
				labelSet[id2] = labelSet[id2] + 1;
				continue;
			}
		}
		closefile(readin);
		cout << "papers_and_labels got!\n";

		openfile(PT_filePath, readin);
		while (readin >> id1 >> id2)
		{
			papers_and_terms[id1].insert(id2);
			terms_and_papers[id2].insert(id1);
			termSet.insert(id2);
			readin >> id1;
		}
		closefile(readin);
		cout << "papers_and_terms got!\n";

	}

};

struct Bayes
{
	//int id1：label, int id2：word, double res：prob
	// beta(id1,id2) = res
	map<int, map<int, double> > beta;

	//the probability of the occurrence of each label
	map<int, double> pai;

	//word frequency
	map<int, map<int, int> > wordINum_in_labelJ;
	map<int, int> wordsNum_in_labelJ;

	Datasets* dataset;

	// get the word frequency in constructor
	Bayes(Datasets& dataset)
	{
		this->dataset = &dataset;
		set<int>::iterator trainIter = trainSet.begin();
		while (trainIter != trainSet.end())
		{
			int label = dataset.papers_and_labels[*trainIter];
			if (label == 0)
			{
				trainIter++;
				continue;
			}
			set<int>::iterator wordIter = dataset.papers_and_terms[*trainIter].begin();
			if (wordsNum_in_labelJ.find(label) == wordsNum_in_labelJ.end())
				wordsNum_in_labelJ[label] = 0;
			while (wordIter != dataset.papers_and_terms[*trainIter].end())
			{
				wordsNum_in_labelJ[label]++;
				if (wordINum_in_labelJ.find(label) != wordINum_in_labelJ.end())
					wordINum_in_labelJ[label][*wordIter]++;
				else
					wordINum_in_labelJ[label][*wordIter] = 0;
				wordIter++;
			}
			trainIter++;
		}
		cout << "NaiveBayes data got!\n";
	}

	//train：get the probability of each label(pai[label]) and the probability of the occurrence
	// of each term in each label(beta[label][term])
	void train()
	{
		cout << "NaiveBayes training..." << endl;
		map<int, int>::iterator labelIter = dataset->labelSet.begin();
		int labelNums = 0;
		while (labelIter != dataset->labelSet.end())
		{
			labelNums += labelIter->second;
			set<int>::iterator termIter = dataset->termSet.begin();
			while (termIter != dataset->termSet.end())
			{
				beta[labelIter->first][*termIter]
					= (wordINum_in_labelJ[labelIter->first][*termIter] + 1)
					/ (double)(wordsNum_in_labelJ[labelIter->first] + dataset->termSet.size());
				termIter++;
			}
			labelIter++;
		}
		labelIter = dataset->labelSet.begin();
		while (labelIter != dataset->labelSet.end())
		{
			pai[labelIter->first] = labelIter->second / (double)labelNums;
			labelIter++;
		}
	}

	//according to the pai and beta ,get the most likely label of a paper and the probability of the label 
	//return label + prob
	pair<int,double> checkLabelForOne(int paper)
	{
		map<int, int>::iterator labelIter = dataset->labelSet.begin();
		int bestlabel = -1;
		double maxprob = 1 << 31;
		double allprob = 0;
		while (labelIter != dataset->labelSet.end())
		{
			int label = labelIter->first;
			set<int>::iterator paperTermIter = (dataset->papers_and_terms)[paper].begin();
			double prob = pai[label];
			while (paperTermIter != (dataset->papers_and_terms)[paper].end())
			{
				prob *= beta[label][*paperTermIter];
				paperTermIter++;
			}
			allprob += prob;
			if (prob > maxprob)
			{
				maxprob = prob;
				bestlabel = label;
			}
			labelIter++;

		}
		assert(bestlabel > 0);
		return make_pair(bestlabel, maxprob/allprob);
	}

	//get the accuracy of NaiveBayes on the test set
	double test()
	{
		set<int>::iterator testSetIter = testSet.begin();
		int trueNum = 0;
		int allNum = 0;
		while (testSetIter != testSet.end())
		{
			allNum++;
			int paper = *testSetIter;
			int testLabel = checkLabelForOne(paper).first;
			int realLabel = dataset->papers_and_labels[paper];
			if (testLabel == realLabel)
				trueNum++;
			testSetIter++;
		}
		double score = trueNum / (double)allNum;
		return score;
	}

	void storeLabelAndProb(string filename)
	{
		map<int, int>::iterator paLaIter = dataset->papers_and_labels.begin();
		ofstream out = ofstream(filename);
		assert(out.is_open());
		while (paLaIter != dataset->papers_and_labels.end())
		{
			pair<int, double>ans = checkLabelForOne(paLaIter->first);
			out << paLaIter->first << "\t" << paLaIter->second
				<< "\t" << ans.first << "\t" << ans.second << "\n";
			paLaIter++;
		}
	}
};

//select testSet and trainSet
void getRandom(Datasets& dataset, int index)
{
	srand(time(NULL));
	testSet.clear();
	trainSet.clear();
	int testNum = DATANUM*index / 100.0;
	for (int i = 0; i < testNum; ++i)
	{
		while (1)
		{
			int id = rand() % 19396 + 1;
			if (testSet.find(id) == testSet.end())
				if (dataset.papers_and_labels[id] != 0)
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

int main()
{
	string filePath1 = "../Preprocessing/paper_labels.txt";
	string filePath2 = "../Cora/PT.txt";
	Datasets dataset = Datasets(filePath1, filePath2);
	string outfile = "";
#ifdef TEST
	double score[25] = { 0 };
	outfile = "NaiveBayes_score.txt"
	for (int i = 20; i <= 40; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			getRandom(dataset, i);
			Bayes bayes = Bayes(dataset);
			bayes.train();
			score[i - 20] += bayes.test();
		}
		cout << "test: " << i << ' ' << "score: " << score[i - 20] / 5 << endl;
	}
	ofstream out = ofstream(outfile);
	for (int i = 0; i <= 20; ++i)
	{
		out << i + 20 << "\t" << score[i]\5 << endl;
	}
	out.close();
#endif

#ifdef GETPROB
	string trainFile = "train.txt";
	string testFile = "test.txt";
	outfile = "paper_real_label_prob.txt"
	ifstream in = ifstream(trainFile);
	assert(in.is_open());
	int id;
	while (in >> id)
	{
		trainSet.insert(id);
	}
	in = ifstream(testFile);
	assert(in.is_open());
	while (in >> id)
	{
		testSet.insert(id);
	}
	Bayes bayes = Bayes(dataset);
	bayes.train();
	bayes.storeLabelAndProb(outfile);
#endif
}
