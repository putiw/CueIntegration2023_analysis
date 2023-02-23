function [trainedClassifier, validationAccuracy] = splitcenter(trainingData)

X = trainingData(:,1:end-1);
Y = trainingData(:,end);
nTrial = size(X,1);
pick = nchoosek(1:10,2);
pick(:,2) = pick(:,2) + 10;

acc = zeros(size(pick,1),1);
for iRep = 1:size(pick,1)

    whichTrain = ~ismember(1:nTrial,pick(iRep,:));
    
    diff = nanmean(X(whichTrain(1:nTrial/2-1),:))-nanmean(X(whichTrain(nTrial/2:end),:))/mean([std(X(whichTrain(1:nTrial/2-1),:)) std(X(whichTrain(nTrial/2:end),:))]);
    
    XX = X(whichTrain,diff>prctile(diff,90));
    X(pick(iRep,:),diff>prctile(diff,90))
    
    
end











