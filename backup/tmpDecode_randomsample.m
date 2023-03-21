% Load data
close all
whichRoi = 8; %5 FST and MST
iCue = 4; % comb
X = betas(roimask{whichRoi},label(:,5)==iCue)';
Y = label(label(:,5)==iCue,3);

aTmp = [X Y];
% Divide data into 10 folds
cv = cvpartition(Y,'KFold',20);

% Initialize variables to store accuracy
acc = zeros(cv.NumTestSets,1);
allver = 1:size(X,2);
% Loop through folds for cross-validation
for i = 1:cv.NumTestSets
    % Get training and testing data for this fold
    trIdx = cv.training(i);
    teIdx = cv.test(i);
    X_train = X(trIdx,:);
    Y_train = Y(trIdx);
    X_test = X(teIdx,:);
    Y_test = Y(teIdx);
    
    %% feature selection
    
    %within this training set, find out which vertices have very low SNR
    %and set them to nan
    
    diff = abs((nanmean(X_train(Y_train==1,:))-nanmean(X_train(Y_train==2,:))))./(nanstd(X_train(Y_train==1,:))*0.5+nanstd(X_train(Y_train==2,:))*0.5);
    [vec ind] = maxk(diff,50);
    X_train(:,allver(~ismember(allver,ind)))=[];
    X_test(:,allver(~ismember(allver,ind)))=[];

    %Train SVM classifier
    %svmModel = fitcsvm(X_train,Y_train,'KernelFunction','rbf');
    
    Model = fitcsvm(...
    X_train, ...
    Y_train, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 40, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [single(1); single(2)]);
    
    % Predict labels for test data
    Y_pred = predict(Model,X_test);
    
    % Calculate accuracy for this fold
    acc(i) = mean(Y_pred == Y_test);
end
mean(acc)
% Visualize decoding accuracy
figure;
bar(acc);
ylim([0 1]);
ylabel('Decoding Accuracy');
xlabel('Fold Number');
title('SVM Classification Accuracy');