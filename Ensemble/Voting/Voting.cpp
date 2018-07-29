/*
  This file is the voting file
  Read in the label from the test set
  Read in the former classifier from the other file
  Implement the voting strategy
*/
#include <cmath>
#include <algorithm>
#include "Nodes.h"
#define NODE_NUM 19396
#define AUTHOR 0
#define CITATION 1
#define NAIVEBAYES 2

// Voting Function
int voting(Node* node, double param0, double param1, double param2) {
	// Achieve the agreement with CRF, as it's most accurate
	if (node->pred_label[1] == node->pred_label[2] || node->pred_label[0] == node->pred_label[1])
		return node->pred_label[1];
	
	// Not achieve agreement
	// Average the weights according to parameter param0, param1, param2
	// Chosse the most likely one
	double tmp[10]; memset(tmp, 0, sizeof(double)*10);
	for (int i = 0; i < 10; i++) 
		tmp[i] = node->likelihood[0][i]*param0+node->likelihood[1][i]*param1+node->likelihood[2][i]*param2;
	double max_likelihood = 0; int index = 0;
	for (int i = 0; i < 10; i++)
		if (tmp[i] > max_likelihood) {
			max_likelihood = tmp[i];
			index = i + 1;
		}
	return index;
}

int main()
{
	char file_path_test[100] = "..//test.txt";
	int test_set[5000]; int total_test = 4849;
	read_test(test_set, file_path_test, total_test);
	// First reading in
	// And make them into an array
	char file_path_aut[100] = "..//Author_res.txt";
	Node *nodes = new Node[19396];
	int total = 0;
	total = read_in(nodes, file_path_aut, NODE_NUM, AUTHOR);
	char file_path_cit[100] = "..//Citation_res.txt";
	total = read_in(nodes, file_path_cit, NODE_NUM, CITATION);
	char file_path_nb[100] = "..//NaiveBayes.txt";
	total = read_in(nodes, file_path_nb, NODE_NUM, NAIVEBAYES);
	
	// Find the nodes in test set
	for (int i = 0; i < total; i++) {
		for (int j = 0; j < total_test; j++) {
			if (test_set[j] == nodes[i].id) {
				nodes[i].if_test = 1;
				break;
			}
			else if (test_set[j] > nodes[i].id) {
				break;
			}
		}
	}
	// Vote for every one and test the accuracy
	int count = 0;
	Node *ptr = nodes;
	for (int i = 0; i < total; i++) {
		int vote_label = voting(ptr, 0.2, 1, 0.9);
		ptr->vote_label = vote_label;
		if (ptr->if_test == 1) {
			//printf("id:%d ,pl:%d %d, rl:%d, vl:%d\n", ptr->id, ptr->pred_label[0], ptr->pred_label[1], ptr->real_label, ptr->vote_label);
			if (ptr->vote_label == ptr->real_label)
				count++;
		}
		ptr++;
	}
	// Print the accuracy
	double accu = (double)count/(double)total_test;
	printf("%d right judgement in %d instances, accuracy is %lf\n", count, total_test, accu);
	/*
	ptr = nodes;
	for (int i = 0; i < 100; i++) {
		printf("%d %d %d %d\n", ptr->id, ptr->real_label, ptr->pred_label[0], ptr->pred_label[1], ptr->vote_label);
	}
	*/
	return 0;
}
