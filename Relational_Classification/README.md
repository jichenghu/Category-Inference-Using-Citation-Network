# Relational Classification
This part of code is used to gather information from cited links (or authors). To run the code, you've to download the UGM toolkit https://www.cs.ubc.ca/~schmidtm/Software/UGM.html. The update of *UGM_CRF_NLL_HiddenC.zip* and *UGM_getEdges.m* in 2013 is required. The update of *UGM_CRF_NLL_HiddenC.zip* in 2015 is recommended. In our experiment, *Cora_TrainCRF.m* can reach an accuracy of about 72% when 25% of papers are used as test set.  

  * *Cora_TrainCRF.m* contains the steps of building model, training, decoding, and inference of a CRF model.
  * *CoraAuthor_TrainCRF.m* is similar to the above one, except that it use co-authors to build edges instead of cited links.
  * *results/* contains the results of conditional inference given testSet in Preprocess folder.
  * *trained_weights/* contains the weights trained given testSet in Preprocess folder.
