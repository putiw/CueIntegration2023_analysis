clear all;close all;clc; 
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct,param] = init_decode_ses2;
% Define the sampling rate (TR) and data
TR = 1.5;
% Define number of repetitions
num_reps = 500;
tic
acc = zeros(num_reps,6,9);
%%
% Specify the number of features to select using mRMR
k = 180;
% Define number of testing trials
num_test_trials = 1;
for iROI = 1:9
data = datafiles{3}(roimask{iROI},:)'; % replace with your own data

%% Compute the power spectrum
[P,f] = pwelch(data,[],[],[],1/TR);

% Plot the power spectrum
semilogy(f,P);
xlabel('Frequency (Hz)');
ylabel('Power');

%% High-pass frequency
% Define the cutoff frequency (in Hz)
cutoff_freq = 0.025; 

% Define the filter parameters
nyquist_freq = 1/(2*TR);
filter_order = 2;
Wn = cutoff_freq/nyquist_freq;

% Create the high-pass filter
[b,a] = butter(filter_order,Wn,'high');

% Apply the filter to the data
filtered_data = filtfilt(b,a,data);
%% Regress out noise
tmpregressor = '/Users/pw1246/Desktop/MRI/CueIntegration2023/derivatives/fmriprep/sub-0248/ses-02/func/sub-0248_ses-02_task-Cue_run-3_desc-confounds_timeseries.csv';
tmpregressor = readtable(tmpregressor);
nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x str2double(tmpregressor.trans_x_derivative1) tmpregressor.trans_x_power2 tmpregressor.trans_y str2double(tmpregressor.trans_y_derivative1) tmpregressor.trans_y_power2 tmpregressor.trans_z str2double(tmpregressor.trans_z_derivative1) tmpregressor.trans_z_power2 tmpregressor.rot_x str2double(tmpregressor.rot_x_derivative1) tmpregressor.rot_x_power2 tmpregressor.rot_y str2double(tmpregressor.rot_y_derivative1) tmpregressor.rot_y_power2 tmpregressor.rot_z str2double(tmpregressor.rot_z_derivative1) tmpregressor.rot_z_power2 tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];        
% calculate the mean time series (global signal)
global_signal = mean(filtered_data, 2);

% create the design matrix for regression
X = [ones(size(global_signal)), global_signal,nuisance];
X(isnan(X))=0;
% perform regression
coefficients = X \ filtered_data;

% calculate the residuals
residuals = filtered_data - X * coefficients;

% overwrite the original data with the residuals
regressed_data = residuals;
%%
dms = dsTrial{3};
dmS = sum(dms,2);

% tmp = zeros(60,size(regressed_data,2),3);
% for iT = 4:6    
%       tmp(:,:,iT-3)= zscore(regressed_data(find(dmS)-1+iT,:));       
% end
% tmp=mean(tmp,3);

tmp = zeros(60,size(regressed_data,2),4);
for iT = 4:7  
      tmp(:,:,iT-3)= zscore(regressed_data(find(dmS)-1+iT,:));       
end
tmp=mean(tmp,3);


lb = nonzeros(dsCon{3}(:,1)+dsCon{3}(:,2)*2);

%% Define data dimensions
num_trials = 30;
num_voxels = 250;





% Define number of classes
num_classes = 2;


class1_data = tmp(lb==1,:);
class2_data =  tmp(lb==2,:);
       
    
    
for rep = 1:num_reps
    [iROI rep]
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



% Compute the relevance scores (correlation between each feature and the labels)
rel_scores = zeros(size(X, 2), 1);
for i = 1:size(X, 2)
    rel_scores(i) = corr(X(:, i), y);
end

% Compute the redundancy scores (average correlation between each feature and
% the top k-1 most relevant features)
red_scores = mean(abs(corr(X)))';


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
    acc(rep,1,iROI) = mean(svm_predicted_labels == test_labels);
    
    % Train and test KNN classifier
    knn_model = fitcknn(train_data, train_labels);
    knn_predicted_labels = predict(knn_model, test_data);
    acc(rep,2,iROI) = mean(knn_predicted_labels == test_labels);
    
    
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
    acc(rep,3,iROI) = mean(knn_predicted_labels == test_labels);
    
    
    classificationKNN = fitcknn(...
    train_data, ...
    train_labels, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);
    knn_predicted_labels = predict(classificationKNN, test_data);
    acc(rep,4,iROI) = mean(knn_predicted_labels == test_labels);
    
    
    classificationKNN = fitcknn(...
    train_data, ...
    train_labels, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);
    knn_predicted_labels = predict(classificationKNN, test_data);
    acc(rep,5,iROI) = mean(knn_predicted_labels == test_labels);
    
     svm_model = fitcsvm(...
    train_data, ...
    train_labels, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 22, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);

    svm_predicted_labels = predict(svm_model, test_data);
    
    acc(rep,6,iROI) = mean(svm_predicted_labels == test_labels);

end

% Report average decoding accuracies
fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(acc(:,1,iROI))*100);
fprintf('Average Euclidean KNN decoding accuracy: %.2f%%\n', mean(acc(:,2,iROI))*100);
fprintf('Average Minkowski KNN decoding accuracy: %.2f%%\n', mean(acc(:,3,iROI))*100);
fprintf('Average weight Euclidean KNN decoding accuracy: %.2f%%\n', mean(acc(:,4,iROI))*100);
fprintf('Average chebychev KNN decoding accuracy: %.2f%%\n', mean(acc(:,5,iROI))*100);
fprintf('Average gaussian SVM decoding accuracy: %.2f%%\n', mean(acc(:,6,iROI))*100);

end
reshape(mean(acc),6,9)
toc