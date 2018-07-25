this commit includes:
 cora dataset
 label_map.txt: a map from original label(0~80) to the upper-class label(0~10)
 UGM_makeCRFmaps.m: downloaded from homepage of UGM
 UGM_makeMRFmaps.m: downloaded from homepage of UGM
 Cora_UGM_TrainCRF.m(not finished yet): adapted from example_UGM_TrainCRF,using loopy belief propagation

The main problem I face now is the loopy belief propagation doesn't converge, If you get a good result, please
save the weight and inform me.Thanks!