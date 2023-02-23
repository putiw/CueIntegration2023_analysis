% Load fMRI data and labels
fmri_data = tmp;
fmri_labels = lb;
cv = cvpartition(length(fmri_labels), 'LeaveOut');

% Define the number of features to select
num_features = 50;




% Initialize the selected feature set to an empty vector
selected_features = [];

% Perform sequential feature selection with backward elimination
for i = 1:num_features
    % Define the set of candidate features as the remaining features that have not yet been selected
    candidate_features = setdiff(1:size(fmri_data,2), selected_features);
    
    % Initialize the performance metric for each candidate feature
    performance_metric = zeros(length(candidate_features), 1);
    
    % Evaluate the performance of each candidate feature using cross-validation
    for j = 1:length(candidate_features)
        [i j]
        % Select the current set of features
        features = [selected_features, candidate_features(j)];
        
        % Initialize the performance metric for the current feature
        fold_performance = zeros(cv.NumTestSets, 1);
        
        % Train and evaluate the classifier using cross-validation
        for k = 1:cv.NumTestSets
            % Get the indices for the training and test sets for the current fold
            train_indices = cv.training(k);
            test_indices = cv.test(k);
            
            % Train the SVM classifier using the selected features
            svm_classifier = fitcsvm(fmri_data(train_indices, features), fmri_labels(train_indices));
            
            % Evaluate the performance of the classifier on the test set for the current fold
            fold_performance(k) = sum(predict(svm_classifier, fmri_data(test_indices, features)) == fmri_labels(test_indices)) / sum(test_indices);
        end
        
        % Compute the average performance of the classifier over all folds for the current feature
        performance_metric(j) = mean(fold_performance);
    end
    
    % Remove the feature with the smallest contribution to the classifier's performance
    [~, worst_feature_index] = max(performance_metric);
    selected_features = setdiff(selected_features, candidate_features(worst_feature_index));
end

% Train the final SVM classifier using the selected feature set
final_svm_classifier = fitcsvm(fmri_data(:, selected_features), fmri_labels);

% Evaluate the performance of the classifier using the selected feature set on the test set (if desired)
test_data = fmri_data(81:end,selected_features);
test_labels = fmri_labels(81:end);
test_performance = predict(final_svm_classifier, test_data);
test_accuracy = sum(test_performance == test_labels) / length(test_labels);
