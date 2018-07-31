# Category-Inference-Using-Citation-Network

### Authors
We are all students from the school of Electronic Engineering and Computer Science of Peking University. Every member contributed equally and the following names are in alphabetical order.
* Xiuping Cui
* Naiqing Guan(guannq@pku.edu.cn)(pinaggle)
* [Ziqi Pang](https://www.linkedin.com/in/ziqi-pang-8b5992158/)(pangziqi@pku.edu.cn)(xingyueyaohui)
* Pengfei Wang


### Overview
This is a course project on graphical probability model. 
In this project, we inference the catagories of computer science papers using their titles, authors, and cited network. 
To test the performance of our algorithm, we run it on datase [CORA](https://relational.fit.cvut.cz/dataset/CORA). 

### Structure
* *Cora* contains the Cora dataset.
* *Preprocessing* contains the test set and the upper classes of papers.
* *Text_Classification* contains a naive-bayes approach to classify papers using their titles and authors.
* *Relational_Classification* contains a CRF appraoch to classify papers using cited network.
* *Ensemble* contains the approach to combine the above two results together to get a better performance.

### Reference
The implementation of CRF network is based on UGM toolkit by Mark Schmidt. https://www.cs.ubc.ca/~schmidtm/Software/UGM.html
The idea of ensembling comes from *Relational Ensemble Classification* by Christine Preisach and Lars Schmidt-Thieme.
 
Hopefully, our work will be of help to you! Please enjoy!
