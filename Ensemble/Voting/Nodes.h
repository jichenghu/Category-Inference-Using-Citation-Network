/*
  This file gives the main components to be used in Voting
  Including definition and reading in
*/
#include <cstdio>
#include <cstring>
class Node {
public:
	int id;
	int real_label;
	int pred_label[3];
	int vote_label;
	double likelihood[3][10];
	int if_test;

	Node() {
		if_test = 0;
		id = real_label = vote_label =0;
		memset(pred_label, 0, sizeof(int)*3);
		memset(likelihood, 0, sizeof(double)*30);
	}
	Node(int i, int r) {
		id = i; real_label = r;
		vote_label = 0;
		memset(pred_label, 0, sizeof(int)*3);
		memset(likelihood, 0, sizeof(double)*30);
	}
};
// From a file read in the data
// num is the number of the total ids
// order is the place we put in 
// return the paper with labels in the file
int read_in(Node *node, char* file, int num, int order) {
	FILE *f = fopen(file, "r");
	Node *ptr = node;
	double tmp[10]; char no_use; int tmp_rlabel, tmp_plabel, tmp_id;
	int count = 0;
	for (int i = 0; i < num; i++) {
		fscanf(f, "%d,%d,%d,", &tmp_id, &tmp_rlabel, &tmp_plabel);
		if (tmp_rlabel != 0 && (tmp_id == ptr->id || ptr->id == 0)) {
			ptr->pred_label[order] = tmp_plabel;
			ptr->id = tmp_id; ptr->real_label = tmp_rlabel;
			for (int j = 0; j < 10; j++) 
				fscanf(f, "%lf%c", &tmp[j], &no_use);
			for (int j = 0; j < 10; j++) 
				ptr->likelihood[order][j] = tmp[j];
			count++;
			ptr += 1;
		}
		else {
			for (int j = 0; j < 10; j++) 
				fscanf(f, "%lf%c", &tmp[j], &no_use);
		}
	}
	return count;
}
void read_test(int *array, char *path, int num) {
	FILE *f = fopen(path, "r");
	for (int i = 0; i < num; i++)
		fscanf(f, "%d\n", &array[i]); 
}
