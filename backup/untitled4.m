

% Define data dimensions
num_trials = 30;
num_voxels = 200;

% Define number of repetitions
num_reps = 5;

% Define number of testing trials
num_test_trials = 3;

% Define number of classes
num_classes = 2;

% Initialize decoding accuracy arrays
svm_acc = zeros(num_reps, 1);
knn_acc = zeros(num_reps, 1);
acc = zeros(num_reps, 1);

class1_data = tmp(lb==1,:);
class2_data =  tmp(lb==2,:);
       
    
    
for rep = 1:num_reps
    % Generate random indices for testing and training data
    test_indices = sort(randperm(num_trials, num_test_trials));
    train_indices = setdiff(1:num_trials, test_indices);
    test_indices2 = sort(randperm(num_trials, num_test_trials));
    train_indices2 = setdiff(1:num_trials, test_indices2);
    % Separate training and testing data
    train_data = [class1_data(train_indices, :); class2_data(train_indices2, :)];
    train_labels = [ones(num_trials-num_test_trials,1); 2*ones(num_trials-num_test_trials,1)];
    test_data = [class1_data(test_indices, :); class2_data(test_indices2, :)];
    test_labels = [ones(num_test_trials,1); 2*ones(num_test_trials,1)];
    
    
        % Assume X is a matrix of features and y is a vector of class labels.
X = train_data; y = train_labels;

% Specify the number of features to select using mRMR
k = 20;

% Compute the relevance scores (correlation between each feature and the labels)
rel_scores = zeros(size(X, 2), 1);
for i = 1:size(X, 2)
    rel_scores(i) = corr(X(:, i), y);
end

% Compute the redundancy scores (average correlation between each feature and
% the top k-1 most relevant features)
red_scores = zeros(size(X, 2), 1);
for i = 1:size(X, 2)
    red_scores(i) = mean(abs(corr(X(:, i), X(:, [1:i-1 i+1:end]))));
end

% Compute the mRMR scores (relevance minus redundancy)
mRMR_scores = rel_scores - red_scores;

% Select the k features with the highest mRMR scores
[~, selected_features] = sort(mRMR_scores, 'descend');
selected_features = selected_features(1:k);

% Subset the data to include only the selected features
X_selected = X(:, selected_features);

train_data =  train_data(:,selected_features);
test_data =  test_data(:,selected_features);

    
    % Train and test SVM classifier
    svm_model = fitcsvm(train_data, train_labels);
    svm_predicted_labels = predict(svm_model, test_data);
    svm_acc(rep) = mean(svm_predicted_labels == test_labels);
    
    % Train and test KNN classifier
    knn_model = fitcknn(train_data, train_labels);
    knn_predicted_labels = predict(knn_model, test_data);
    knn_acc(rep) = mean(knn_predicted_labels == test_labels);
    
    
    classificationKNN = fitcknn(...
    train_data, ...
    train_labels, ...
    'Distance', 'Minkowski', ...
    'Exponent', 3, ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);
    knn_predicted_labels = predict(classificationKNN, test_data);
    acc(rep,1) = mean(knn_predicted_labels == test_labels);
    
    
    classificationKNN = fitcknn(...
    train_data, ...
    train_labels, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);
    knn_predicted_labels = predict(classificationKNN, test_data);
    acc(rep,2) = mean(knn_predicted_labels == test_labels);
    

end

% Report average decoding accuracies
fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(svm_acc)*100);
fprintf('Average KNN decoding accuracy: %.2f%%\n', mean(knn_acc)*100);
fprintf('Average KNN decoding accuracy: %.2f%%\n', mean(acc(:,1))*100);
fprintf('Average KNN decoding accuracy: %.2f%%\n', mean(acc(:,2))*100);
