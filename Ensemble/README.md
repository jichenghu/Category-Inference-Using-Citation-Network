# Ensemble
### Voting
In this section, we test the forst ensemble method "voting".
Our voting strategy is very simple. We weighted avrage the different possibility distribution from different classifiers and give out the final answer.
The weight of the different classifiers are tuned manually with a validation set. 
We tried two versions of voting implementation. They achieve similar performance.
### NN
This neural network uses the **citation result** and **NaiveBayes result** as input,the actual label as output.This neural network has three hidden layers, using regularization, Adam optimization method, learning rate is 7e-4, and trained 800 rounds.
