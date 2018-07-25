clear all
close all


% Make labels y, and term features X
y = load('..//Cora//paper_label.txt','-ascii');
y = int32(y(:,2)');
[nInstances,nNodes] = size(y);
y_map = load('..//Cora//label_map.txt','-ascii');
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
% treat no label as another label
y = y + 1;


%% Make edgeStruct
nStates = max(y);
adj = zeros(nNodes,nNodes);
edgeTmp = load('..//Cora//PP.txt','ascii');
nEdges = size(edgeTmp,1);
for i = 1:nEdges
    adj(edgeTmp(i,1),edgeTmp(i,2)) = i;
end
adj = adj+adj';   
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nEdges = edgeStruct.nEdges;
maxState = max(nStates);

% check correctness of edgeStruct
% [row,col] = find(adj~=0);
% num = size(row); 
% if (nEdges ~= num/2)
%    fprintf('edgeStruct error\n');
%    return;
% end
%% inferrence using features trained before

%% Training (with node features, but no edge features)
% make term features X using only nFeatures words
Xtmp = load('..//Cora//PT.txt','-ascii');
nFeatures = 100;
Xnode = zeros(nInstances,nFeatures,nNodes);
Xedge = ones(nInstances,1,nEdges);
nWords = size(Xtmp,1);
% we suppose nInstance==1 here for Cora
for i = 1:nWords
    if(Xtmp(i,2) <= nFeatures)
        Xnode(1,Xtmp(i,2),Xtmp(i,1)) = Xnode(1,Xtmp(i,2),Xtmp(i,1))+1;
    end
end        
Xnode = [ones(nInstances,1,nNodes) Xnode];
nNodeFeatures = size(Xnode,2);

% Make nodeMap
nodeMap = zeros(nNodes,maxState,nNodeFeatures,'int32');
cnt = 0;
for f = 1:nNodeFeatures
    for k = 1:maxState-1
        cnt = cnt+1;
        nodeMap(:,k,f) = cnt;
    end
end
nNodeParams = cnt;
% Make edgeMap
edgeMap = zeros(maxState,maxState,nEdges,'int32');
for i = 1:maxState
    for j = 1:maxState
        if( i==maxState && j ==maxState) 
            continue;
        end
        cnt = cnt + 1;
        edgeMap(i,j,:) = cnt;
    end
end

% Initialize weights
nParams = max([nodeMap(:);edgeMap(:)]);
w = rand(nParams,1);

% Optimize
LB = zeros(nParams,1);
LB(1:nNodeParams,1) = -Inf;
UB = repelem(Inf,nParams)';
funObj = @(w)UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
w = minConf_TMP(funObj,w,LB,UB)
fprintf('(paused)\n');
pause

%% Training (no features)

% Make simple bias features
Xnode = ones(nInstances,1,nNodes);
Xedge = ones(nInstances,1,nEdges);

% Make nodeMap
ising = 0; %  use ising approximation or not
tied = 1; % suppose all parameters of edges are same or not
[nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied);
% nodeMap = zeros(nNodes,maxState,'int32');
% nodeMap(:,1) = 1;

% edgeMap = zeros(maxState,maxState,nEdges,'int32');
% edgeMap(1,1,:) = 2;
% edgeMap(2,1,:) = 3;
% edgeMap(1,2,:) = 4;

% Initialize weights
nParams = max([nodeMap(:);edgeMap(:)]);
nNodeParams = max(nodeMap(:));
w = rand(nParams,1)*100;
LB = zeros(nParams,1);
UB = zeros(nParams,1);
LB(1:nNodeParams,:) = -Inf;
UB = repelem(Inf,nParams)';
% Optimize
funObj = @(w)UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
w = minConf_TMP(funObj,w,LB,UB)
% w = minFunc(@UGM_CRF_NLL,randn(size(w)),[],Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Junction)

% Example of making potentials for the first training example
instance = 1;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,instance);
nodePot(1,:)
edgePot(:,:,1);
fprintf('(paused)\n');
pause


%% Training (with edge features)

% Make edge features
sharedFeatures = 1:13;
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);
nEdgeFeatures = size(Xedge,2);

% Make edgeMap
edgeMap = zeros(maxState,maxState,nEdges,nEdgeFeatures,'int32');
for edgeFeat = 1:nEdgeFeatures
    for s1 = 1:2
        for s2 = 1:2
            f = f+1;
            edgeMap(s1,s2,:,edgeFeat) = f;
        end
    end
end

% Initialize weights
nParams = max([nodeMap(:);edgeMap(:)]);
w = zeros(nParams,1);

% Optimize
UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain);
w = minFunc(@UGM_CRF_NLL,w,[],Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain)
fprintf('(paused)\n');
pause

%% Do decoding/infence/sampling in learned model (given features)

% We will look at a case in December
i = 11;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);

decode = UGM_Decode_Chain(nodePot,edgePot,edgeStruct)

[nodeBel,edgeBel,logZ] = UGM_Infer_Chain(nodePot,edgePot,edgeStruct);
nodeBel

samples = UGM_Sample_Chain(nodePot,edgePot,edgeStruct);
figure(1);
imagesc(samples')
title('Samples from CRF model (for December)');
fprintf('(paused)\n');
pause

%% Do conditional decoding/inference/sampling in learned model (given features)

clamped = zeros(nNodes,1);
clamped(1:2) = 2;

condDecode = UGM_Decode_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Decode_Chain)
condNodeBel = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Infer_Chain)
condSamples = UGM_Sample_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Sample_Chain);

figure(2);
imagesc(condSamples')
title('Conditional samples from CRF model (for December)');
fprintf('(paused)\n');
pause

%% Now see what samples in July look like

XtestNode = [1 0 0 0 0 0 0 1 0 0 0 0 0]; % Turn on bias and indicator variable for July
XtestNode = repmat(XtestNode,[1 1 nNodes]);
XtestEdge = UGM_makeEdgeFeatures(XtestNode,edgeStruct.edgeEnds,sharedFeatures);

[nodePot,edgePot] = UGM_CRF_makePotentials(w,XtestNode,XtestEdge,nodeMap,edgeMap,edgeStruct);

samples = UGM_Sample_Chain(nodePot,edgePot,edgeStruct);
figure(3);
imagesc(samples')
title('Samples from CRF model (for July)');
fprintf('(paused)\n');
pause

%% Training with L2-regularization

% Set up regularization parameters
lambda = 10*ones(size(w));
lambda(1) = 0; % Don't penalize node bias variable
lambda(14:17) = 0; % Don't penalize edge bias variable
regFunObj = @(w)penalizedL2(w,@UGM_CRF_NLL,lambda,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain);

% Optimize
w = zeros(nParams,1);
w = minFunc(regFunObj,w);
NLL = UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain)
