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
fprintf('the first %d labels are:\n',disp_len);
y(1:disp_len)

%% Make edgeStruct
nStates = max(y);
adj_single = zeros(nNodes,nNodes);
edgeTmp = load('PP.txt','ascii');
nEdges = size(edgeTmp,1);
for i = 1:nEdges
    adj_single(edgeTmp(i,1),edgeTmp(i,2)) = 1;
end
adj = adj_single+adj_single';   
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nEdges = edgeStruct.nEdges;
maxState = max(nStates);
fprintf('the first %d edges are:\n',disp_len);
edgeStruct.edgeEnds(1:disp_len,:);

% check what the edges looks like
same = 0;
diff = 0;
for i = 1:nEdges
    if(y(edgeStruct.edgeEnds(i,1))==y(edgeStruct.edgeEnds(i,2)))
        same = same+1;
    else
        diff = diff+1;
    end
end
fprintf('%d edges connect same labels, %d edges connect different labels\n',same,diff);

%% make term features X
Xtmp = load('PT.txt','-ascii');
% HINT: nFeatures decides how many words will be used. You can simply
% change it to include more or less words (even 0).
nFeatures = 100; 
Xnode = zeros(nInstances,nFeatures,nNodes);
Xedge = ones(nInstances,1,nEdges);
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
edgeMap = zeros(maxState,maxState,nEdges,'int32');
for i = 1:maxState
    for j = 1:maxState
        cnt = cnt + 1;
        edgeMap(i,j,:) = cnt;
    end
end
fprintf('%d parameters for nodes, %d parameters for edges\n',nNodeParams,cnt-nNodeParams);
fprintf('(paused)\n');
pause

%% Training 
% Initialize weights
nParams = max([nodeMap(:);edgeMap(:)]);
w = zeros(nParams,1);
% Optimize
funObj = @(w)UGM_CRF_NLL_Hidden(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
options = [];
options.method = 'qnewton';
options.MaxIter = 100;
options.LS = 2; % choose Line Search type 
options.LS_init = 2; % Choose step length
options.To1X = 1e-9; % Choose termination tolerance

% HINT: if you want to use a trained feature to do inference directly, then
% comment the minFunc and save command, and use load instead.
% w = minFunc(funObj,w,options);
save('w_100f_2e.mat','w');
% load('w_100f_qs.mat');
fprintf('(paused)\n');
pause

%% Do decoding/infence in learned model (given features)

[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
% get marginals for each edge and node
[nodeBelLBP,edgeBelLBP,logZ] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
% nodeBelLBP(1:100,:)
% obtain the maximum of marginals
maxOfMarginalsLBPdecode = UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
correct_num = sum(maxOfMarginalsLBPdecode == y');
fprintf('Marginal:%d of %d labels are correct,correct rate is %f\n',correct_num,nNodes,correct_num/nNodes);
% obtain MAP inference
decodeLBP = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
correct_num = sum(decodeLBP == y');
fprintf('MAP:%d of %d labels are correct,correct rate is %f\n',correct_num,nNodes,correct_num/nNodes);
fprintf('(paused)\n');
pause

%% Do conditional decoding/inference/sampling in learned model (given features)
nRepeat = 1;
for kase = 1:nRepeat
    clamped = y'; % nNodes * 1
    nTestSet = 1000;
    test = zeros(nTestSet,1);
    cnt = 0;
    while (cnt < nTestSet)    
        ind = int32(rand()*nNodes);
        if(ind == 0) 
            ind = ind + 1;
        end
        if(y(ind) > 0)
            cnt = cnt + 1;
            test(cnt,1) = ind;
            clamped(ind,1) = 0;
        end
    end 
    % get conditional marginals for node end edge
    [condnodeBel,condedgeBel,logZ] = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Infer_LBP);
    % condnodeBel(1:100,:)
    correct_num = 0;
    [~,maxPos] = max(condnodeBel,[],2);
    for i = 1:nTestSet
        if(maxPos(test(i)) == y(test(i)))
            correct_num = correct_num + 1;
        end
    end
    fprintf('Cond Marginal:%d of %d labels are correct,correct rate is %f\n',correct_num,nTestSet,correct_num/nTestSet);
    % conditional decode process
    condDecode = UGM_Decode_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Decode_LBP);
    correct_num = 0;
    for i = 1:nTestSet
        if(condDecode(test(i)) == y(test(i)))
            correct_num = correct_num + 1;
        end
    end
    fprintf('Cond MAP:%d of %d labels are correct,correct rate is %f\n\n',correct_num,nTestSet,correct_num/nTestSet);
end
