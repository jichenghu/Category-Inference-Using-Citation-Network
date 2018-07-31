#include<fstream>
#include<iostream>
#include<string>
#include<map>
#include<set>
#include<vector>
#include<cmath>
#include<assert.h>
#include<iterator>
using namespace std;

set<int> testSet;
void get_testSet(string Path,string test)
{
	string file = Path + "/" + test;
	ifstream in = ifstream(file);
	assert(in.is_open());
	int paper;
	while (in >> paper)
	{
		testSet.insert(paper);
	}
	in.close();
}
map<int, int> paper_and_label;
//需要文件路径
//Bayes的结果
//PT.txt
struct Bayes
{
	vector<vector<double> > probs;
	map<int, int> term_num;
	Bayes(string Path, string Bayes_result,string terms,string author)
	{
		string bayes = Path + '/' + Bayes_result;
		ifstream in = ifstream(bayes);
		assert(in.is_open());
		vector<double> tmp;
		probs.push_back(tmp);
		int paper;
		int label;
		while (in >> paper)
		{
			char de;
			in >> de >> label;
			paper_and_label[paper] = label;
			in >> de >> label;
			vector<double> prob;
			for (int i = 0; i < 10; ++i)
			{
				double pro = 0;
				in >> de >> pro;
				prob.push_back(pro);
			}
			probs.push_back(prob);
		}
		in.close();
		int term;
		in = ifstream(Path + "/" + terms);
		assert(in.is_open());
		while (in >> paper >> term >> label)
		{
			term_num[paper] = term_num[paper] + 1;
		}
		in.close();
		in = ifstream(Path + "/" + author);
		assert(in.is_open());
		while (in >> paper >> term >> label)
		{
			term_num[paper] = term_num[paper] + 1;
		}
		in.close();
	}
};

struct Marcov
{
	vector<vector<double> > probs;
	map<int, int> ci_num;
	Marcov(string Path, string Marcov_result, string cits)
	{
		string marcov = Path + '/' + Marcov_result;
		ifstream in = ifstream(marcov);
		assert(in.is_open());
		vector<double> tmp;
		probs.push_back(tmp);
		int paper;
		int label;
		while (in >> paper)
		{
			char de;
			in >> de >> label;
			paper_and_label[paper] = label;
			in >> de >> label;
			vector<double> prob;
			for (int i = 0; i < 10; ++i)
			{
				double pro;
				in >> de >> pro;
				prob.push_back(pro);
			}
			probs.push_back(prob);
		}
		in.close();
		int cit;
		in = ifstream(Path + "/" + cits);
		assert(in.is_open());
		while (in >> paper >> cit >> label)
		{
			ci_num[paper] = ci_num[paper] + 1;
			ci_num[cit] = ci_num[cit] + 1;
		}
		in.close();
	}
};

int main()
{
	string filePath;
  cout<<"file_Path : ";
  cin>>filePath;
	string test = "Sets-10%/test2.txt";
  cout<<"testSet_name : ";
  cin>>test;
	string bayes_result = "res/BAYES_10_2.csv";
  cout<<"Naivebayes_res : ";
  cin>>bayes_result;
	string marcov_result = "res/CRF_PRLP_10_2.csv";
  cout<<"Naivebayes_res : ";
  cin>>marcov_result;
	string PT = "../Cora/PT.txt";
	string PP = "../Cora/PP.txt";
	string PA = "../Cora/PA.txt";
	get_testSet(filePath, test);
	Bayes bayes = Bayes(filePath, bayes_result, PT, PA);
	Marcov marcov = Marcov(filePath, marcov_result, PP);
	double rate1, rate2;
	cout << "Please (Naivebayes)rate1 and (Marcov)rate2 : ";
	while (cin >> rate1 >> rate2)
	{
		set<int>::iterator setIter = testSet.begin();
		int correct_num = 0;
		while (setIter != testSet.end())
		{
			int paper = *setIter;
			int label = paper_and_label[paper];
			double term_num = rate1;//*bayes.term_num[paper];
			double cit_num = rate2;//*marcov.ci_num[paper];
			double maxprob = 0.0;
			int bestlabel = -1;
			for (int i = 0; i < 10; ++i)
			{
				double prob = term_num*bayes.probs[paper][i]
					+ cit_num*marcov.probs[paper][i];
				if (prob > maxprob)
				{
					maxprob = prob;
					bestlabel = i + 1;
				}
			}
			if (bestlabel == label)
				correct_num++;
			setIter++;
		}
		cout << "accuracy score:" << correct_num / (double)testSet.size() << endl;
	}
}
