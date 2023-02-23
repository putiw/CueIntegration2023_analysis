clear all;close all;clc;
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct,param] = init_decode_ses3;

%% params

TR = 1.5;
whichRoi = 3; % use V3A as example

%% step 1 - For each ROI, combine and reorganize data 

data = cellfun(@(m)m(roimask{whichRoi},:), datafiles, 'UniformOutput', 0);
data = cat(2,data{:})';

% time series 1st to 10th TR after each onsets
figure(1);clf
showtimeseries(datafiles,dsCon,roimask{whichRoi},10,[1 2],[-1.5 1.5],[])

%% step 2 - High-pass filter

figure(2);clf
subplot(1,2,1)
[P,f] = pwelch(data,[],[],[],1/TR);
% Plot the power spectrum
semilogy(f,P);
xlabel('Frequency (Hz)');
ylabel('Power');
ylim([1e0 1e10])

cutoff_freq = 0.025;  %0.025
% Define the filter parameters
nyquist_freq = 1/(2*TR);
filter_order = 2;
Wn = cutoff_freq/nyquist_freq;
% Create the high-pass filter
[b,a] = butter(filter_order,Wn,'high');
% Apply the filter to the data
filtered_data = filtfilt(b,a,data);
[P,f] = pwelch(filtered_data,[],[],[],1/TR);
% Plot the power spectrum
subplot(1,2,2)
semilogy(f,P);
xlabel('Frequency (Hz)');
ylabel('Power');
ylim([1e0 1e10])

%% step 3 - Regress out noise

nuisance = [];
for iRun = 1:4
    tmpregressor = ['/Users/pw1246/Desktop/MRI/CueIntegration2023/derivatives/fmriprep/sub-0248/ses-03/func/sub-0248_ses-03_task-Cue_run-' num2str(iRun) '_desc-confounds_timeseries.csv'];
    tmpregressor = readtable(tmpregressor);
    tmp = [tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x str2double(tmpregressor.trans_x_derivative1) tmpregressor.trans_x_power2 tmpregressor.trans_y str2double(tmpregressor.trans_y_derivative1) tmpregressor.trans_y_power2 tmpregressor.trans_z str2double(tmpregressor.trans_z_derivative1) tmpregressor.trans_z_power2 tmpregressor.rot_x str2double(tmpregressor.rot_x_derivative1) tmpregressor.rot_x_power2 tmpregressor.rot_y str2double(tmpregressor.rot_y_derivative1) tmpregressor.rot_y_power2 tmpregressor.rot_z str2double(tmpregressor.rot_z_derivative1) tmpregressor.rot_z_power2 tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];
    nuisance = [nuisance;tmp];
end
% calculate the mean time series (global signal)
global_signal = mean(filtered_data, 2);

% create the design matrix for regression
X = [ones(size(global_signal)), global_signal,nuisance];
X(isnan(X))=0;
% perform regression
coefficients = X \ filtered_data;

% calculate the residuals
regressed_data = filtered_data - X * coefficients;


%% step 4 - Normalize nth TR

dsT = cat(1,dsTrial{:});
designMat = sum(dsT,2);

tmp = zeros(sum(designMat),size(regressed_data,2),4);
for whichTR = 4:7
    tmp(:,:,whichTR-3)= zscore(regressed_data(find(designMat)-1+whichTR,:));
end
tmp=mean(tmp,3);

%% step 5 - MRMR
k = 200;
% Define number of testing trials
num_test_trials = 1;
num_reps = 100;
acc = zeros(num_reps,3,9);
%
lb = label(:,3);
% Define data dimensions
num_trials = 176/2;
num_voxels = 250;

% Define number of classes
num_classes = 2;


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
    
    
    
    
    % Compute the relevance scores
    % (correlation between each feature and the labels)
    relevance = zeros(size(train_data, 2), 1);
    for i = 1:size(train_data, 2)
        relevance(i) = corr(train_data(:, i), train_labels);
    end
    
    % Compute the redundancy scores
    % (average correlation between each feature and
    % the top k-1 most relevant features)
    redundancy = mean(abs(corr(train_data)))';
    
    % Compute the mRMR scores (relevance minus redundancy)
    MRMR = relevance - redundancy;
    
    % Select the k features with the highest mRMR scores
    [~, whichFeatures] = sort(MRMR, 'descend');
    whichFeatures = whichFeatures(1:k);
    
    % Subset the data to include only the selected features
    train_data =  train_data(:,whichFeatures);
    test_data =  test_data(:,whichFeatures);
    
    
    % Train and test SVM classifier
    svm_model = fitcsvm(train_data, train_labels);
    
    svm_predicted_labels = predict(svm_model, test_data);
    acc(rep,1,whichRoi) = mean(svm_predicted_labels == test_labels);
    
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
    
    acc(rep,2,whichRoi) = mean(svm_predicted_labels == test_labels);
    
    classificationKNN = fitcknn(...
        train_data, ...
        train_labels, ...
        'Distance', 'Minkowski', ...
        'Exponent', 3, ...
        'NumNeighbors', 88, ...
        'DistanceWeight', 'Equal', ...
        'Standardize', false, ...
        'ClassNames', [1; 2]);
    
    
    svm_predicted_labels = predict(classificationKNN, test_data);
    
    acc(rep,3,whichRoi) = mean(svm_predicted_labels == test_labels);
    
    reshape(mean(acc(1:rep,:,:),1),3,9)
end
    
    % Report average decoding accuracies
    fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(acc(:,1,whichRoi))*100);
    fprintf('Average gaussian SVM decoding accuracy: %.2f%%\n', mean(acc(:,2,whichRoi))*100);