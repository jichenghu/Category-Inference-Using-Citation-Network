# Relational Classification
In this section, we will do the relation classification. It contaions the following steps:
* Make the files into a graph (Optional)
  * *elements.h* contains the basic components of the graph *i.e.* vertice and edge
  * *make_graph.cpp* contains the functions to construct structures of the graph
  * *test_graph.cpp* is for manually check the correctness of the graph, youcan play freely!
* Train a conditional random field models using cited links and authors
  * *Cora_TrainCRF.m* contains the steps of building model, training, decoding, and inference
  * *CoraAuthor_TrainCRF.m* is similar to the above one, except that it use co-authors to build edges instead of cited links
  * *Cora/label_map.txt* is used to map specific classes into more general ones
  * *results/* contains the results of conditional inference given testSet
  * *trained_weights/* contains the weights trained given testset
