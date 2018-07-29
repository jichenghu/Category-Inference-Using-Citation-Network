% This code is used to train a Conditional Random Field on Cora dataset to 
% predict the catagories of papers. 
% Author : Gnaiqing
% TODO LIST:
% change the edgeMap to consider different kinds of edges
% use the mex version of CRF_NLL_HIDDEN to accerlerate the inference
% include more words and author feature (which can be very slow if mex is
% not used )

clear all
close all


%% Make labels y

y = load('paper_label.txt','-ascii');
y = int32(y(:,2)');
[nInstances,nNodes] = size(y);
y_map = load('label_map.txt','-ascii');
mapLength = size(y_map,1);
% map the labels to the top-class ones
for i = 1:nInstances
    for j = 1:nNodes
        for k = 1:mapLength
            if(y(i,j) == y_map(k,1))
                y(i,j) = y_map(k,2);
                break;
            end
        end
    end
end
disp_len = 20;
fprintf('the first real %d labels are:\n',disp_len);
y(1:disp_len)
% hide some labels for test, and use the result of naive-bayes to predict
% some labels
y_origin = y;


test = load('test.txt','-ascii');
nTestSet = size(test,1);
for i = 1:nTestSet
    y(test(i)) = 0;
end
% nTestSet = 4849; % use 25% percent of labels as test set
% test = zeros(nTestSet,1);
% cnt = 0;
% while (cnt < nTestSet)    
%     ind = int32(rand()*nNodes);
%     if(ind == 0) 
%         ind = ind + 1;
%     end
%     if(y(ind) > 0)
%         cnt = cnt + 1;
%         test(cnt,1) = ind;
%         y(ind) = 0;
%     end
% end 
% Pred = load('paper_real_label_prob.txt','-ascii');
% threshold = 0.9;
% for i = 1:nNodes
%     if(Pred(i,2)==0 && Pred(i,4)>threshold)
%         % y(Pred(i,1)) = int32(Pred(i,3));
%     end
% end
fprintf('the first %d labels for training are:\n',disp_len);
y(1:disp_len)

%% Make edgeStruct
nStates = max(y);
PA = load('PA.txt','ascii');
nEdges = size(PA,1); 
nAuthors = max(PA(:,2));
adj_PA = zeros(nNodes,nAuthors);
for i = 1:nEdges
    adj_PA(PA(i,1),PA(i,2)) = 1;
end
nWorks = sum(adj_PA);
adj = zeros(nNodes,nNodes);
for i = 1:nAuthors
    nodeList = find(adj_PA(:,i)~=0);
    len = length(nodeList);
    for j = 1:len-1
        adj(nodeList(j),nodeList(j+1)) = 1;
        adj(nodeList(j+1),nodeList(j)) = 1;
    end
%     for j = 1:len
%         for k = j+1:len
%             adj(nodeList(j),nodeList(k)) = 1;
%             adj(nodeList(k),nodeList(j)) = 1;
%         end
%     end
end
% need to change
% adj(1,:) = 1;
% adj(:,1) = 1;
% adj(1,1) = 0;

neighbors = sum(adj);
nGroups = 10;
nWidth = 3;
data = zeros(1,nGroups);
for i = 1:nGroups-1
    data(i) = sum((neighbors>=(i-1)*nWidth)&(neighbors<i*nWidth));
end
data(nGroups) = sum(neighbors >= (nGroups-1)*nWidth);
minX = 0:nWidth:(nGroups-1)*nWidth;
figure(1);
bar(minX,data);
title('size of neighbors');

data = zeros(1,nGroups);
for i = 1:nGroups-1
    data(i) = sum((nWorks>=(i-1)*nWidth)&(nWorks<i*nWidth));
end
data(nGroups) = sum(nWorks >= (nGroups-1)*nWidth);
minX = 0:nWidth:(nGroups-1)*nWidth;
figure(2);
bar(minX,data);
title('number of papers');

edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nEdges = edgeStruct.nEdges;
maxState = max(nStates);
fprintf('%d Edges in total, the first %d edges are:\n',nEdges,disp_len);
edgeStruct.edgeEnds(1:disp_len,:)
fprintf('(paused)');
pause

%% make term features and edge features 
Xtmp = load('PT.txt','-ascii');
% HINT: nFeatures decides how many words will be used. You can simply
% change it to include more or less words (even 0).
nFeatures = 0; 
Xnode = zeros(nInstances,nFeatures,nNodes);

nWords = size(Xtmp,1);
for i = 1:nWords
    if(Xtmp(i,2) <= nFeatures)
        Xnode(1,Xtmp(i,2),Xtmp(i,1)) = Xnode(1,Xtmp(i,2),Xtmp(i,1))+1;
    end
end        
Xnode = [ones(nInstances,1,nNodes) Xnode];
nNodeFeatures = size(Xnode,2);
fprintf('the features for the first %d terms are:\n',disp_len);
permute(Xnode(1,:,1:disp_len),[3,2,1])

nEdgeFeatures = 1; 
Xedge = zeros(nInstances,nEdgeFeatures,nEdges);
for i = 1:nEdges
    v1 = edgeStruct.edgeEnds(i,1);
    v2 = edgeStruct.edgeEnds(i,2);
    Xedge(1,1,i) = 1;
    if(nEdgeFeatures > 1)
        if(adj_single(v1,v2)~=0)
            Xedge(1,2,i) = 1;
        end
        if(adj_single(v2,v1)~=0)
            Xedge(1,3,i) = 1;
        end
    end
end

%% Make nodeMap and edgeMap
nodeMap = zeros(nNodes,maxState,nNodeFeatures,'int32');
cnt = 0;
for f = 1:nNodeFeatures
    for k = 1:maxState
        cnt = cnt+1;
        nodeMap(:,k,f) = cnt;
    end
end
nNodeParams = cnt;
edgeMap = zeros(maxState,maxState,nEdges,nEdgeFeatures,'int32');
for i = 1:maxState
    for j = i:maxState
        cnt = cnt + 1;
        edgeMap(i,j,:,1) = cnt;
        edgeMap(j,i,:,1) = cnt;
        if(nEdgeFeatures > 1)
            if(i < j)
                cnt = cnt + 1;
                edgeMap(i,j,:,2) = cnt;
                edgeMap(j,i,:,3) = cnt;
                cnt = cnt + 1;
                edgeMap(i,j,:,3) = cnt;
                edgeMap(j,i,:,2) = cnt;
            end
            if (i == j)
                cnt = cnt + 1;
                edgeMap(i,j,:,2) = cnt;
                edgeMap(j,i,:,3) = cnt; 
            end        
        end
    end
end
fprintf('%d parameters for nodes, %d parameters for edges\n',nNodeParams,cnt-nNodeParams);
filename = sprintf('w_Author_%dn_%de.mat',nNodeParams,cnt-nNodeParams);
fprintf('(paused)\n');
pause

%% Training 
% Initialize weights
nParams = max([nodeMap(:);edgeMap(:)]);
w = zeros(nParams,1);
% Optimize
% Set up regularization parameters
% lambda = 10*ones(size(w));
% lambda(1) = 0; % Don't penalize node bias variable
% lambda(14:17) = 0; % Don't penalize edge bias variable
% regFunObj = @(w)penalizedL2(w,@UGM_CRF_NLL,lambda,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain);
funObj = @(w)UGM_CRF_NLL_Hidden(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
options = [];
options.method = 'qnewton';
options.MaxIter = 100;
options.LS = 2; % choose Line Search type 
options.LS_init = 2; % Choose step length
options.To1X = 1e-9; % Choose termination tolerance

% HINT: if you want to use a trained feature to do inference directly, then
% comment the minFunc and save command, and use load instead.
w = minFunc(funObj,w,options);
save(filename,'w');
% load(filename);
fprintf('(paused)\n');
pause

%% Do decoding/infence in learned model (given features)

[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
% get marginals for each edge and node
[nodeBelLBP,edgeBelLBP,logZ] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
% nodeBelLBP(1:100,:)
% obtain the maximum of marginals
maxOfMarginalsLBPdecode = UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
correct_num = sum(maxOfMarginalsLBPdecode == y_origin');
fprintf('Marginal:%d of %d labels are correct,correct rate is %f\n',correct_num,nNodes,correct_num/nNodes);
% obtain MAP inference
decodeLBP = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
correct_num = sum(decodeLBP == y_origin');
fprintf('MAP:%d of %d labels are correct,correct rate is %f\n',correct_num,nNodes,correct_num/nNodes);
fprintf('(paused)\n');
pause

%% Do conditional decoding/inference/sampling in learned model (given features)
fprintf('%d of %d papers are used as test set,the rate is %f\n',nTestSet,nNodes,nTestSet/nNodes);
clamped = y_origin'; % nNodes * 1
for i = 1:nTestSet
    clamped(test(i)) = 0;
end
% get conditional marginals for node end edge
[condnodeBel,condedgeBel,logZ] = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Infer_LBP);
correct_num = 0;
[~,maxPos] = max(condnodeBel,[],2);
for i = 1:nTestSet
    if(maxPos(test(i)) == y_origin(test(i)))
        correct_num = correct_num + 1;
    end
end
fprintf('Cond Marginal:%d of %d labels are correct,correct rate is %f\n',correct_num,nTestSet,correct_num/nTestSet);
% conditional decode process
condDecode = UGM_Decode_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Decode_LBP);
correct_num = 0;
for i = 1:nTestSet
    if(condDecode(test(i)) == y_origin(test(i)))
        correct_num = correct_num + 1;
    end
end
fprintf('Cond MAP:%d of %d labels are correct,correct rate is %f\n\n',correct_num,nTestSet,correct_num/nTestSet);
fprintf('(paused)\n');
pause

%% store the conditional inference result for further use
output = fopen('CRFAuthor_paper_real_label_prob.txt','w');
% prob = [double((1:nNodes)') double(y_origin') double(condDecode  condnodeBel];
for i = 1:nNodes
    fprintf(output,'%d,%d,%d',i,y_origin(i),condDecode(i));
    for j = 1:nStates
        fprintf(output,',%.8f',condnodeBel(i,j));
    end
    fprintf(output,'\n');
end
fclose(output);

