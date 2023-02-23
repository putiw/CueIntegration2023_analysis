clear all;close all;clc; 
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct,param] = init_decode_ses2;
% Define the sampling rate (TR) and data
TR = 1.5;
% Define number of repetitions
num_reps = 50;
tic
acc = zeros(num_reps,5,9);
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
nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x tmpregressor.trans_y tmpregressor.trans_z tmpregressor.rot_x tmpregressor.rot_y tmpregressor.rot_z tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];        

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

tmp = zeros(60,size(regressed_data,2),3);
for iT = 4:6    
      tmp(:,:,iT-3)= zscore(regressed_data(find(dmS)-1+iT,:));       
end
tmp=mean(tmp,3);
lb = nonzeros(dsCon{3}(:,1)+dsCon{3}(:,2)*2);

%% Define data dimensions
num_trials = 30;
num_voxels = 200;



% Define number of testing trials
num_test_trials = 3;

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

%%
numFeatures = 100;
selectedFeatures = [];
    remainingFeatures = 1:size(X, 2);

    % Repeat until the desired number of features is selected
    for i = 1:numFeatures
        % Initialize best feature and best accuracy
        bestFeature = 0;
        bestAcc = 0;

        % Iterate over remaining features
        for j = remainingFeatures
            % Select current feature and previously selected features
            features = [selectedFeatures j];

            % Split data into training and validation sets
            cv = cvpartition(y, 'HoldOut', 0.2);
            dataTrain = X(cv.training, features);
            labelsTrain = y(cv.training);
            dataValid = X(cv.test, features);
            labelsValid = y(cv.test);

            % Fit classifier and compute accuracy
            
    svm_model = fitcsvm(dataTrain, labelsTrain);
    svm_predicted_labels = predict(svm_model, dataValid);
    accc = mean(svm_predicted_labels == labelsValid);
    
            % If accuracy is better than the best so far, update best feature
            if accc > bestAcc
                bestFeature = j;
                bestAcc = accc;
            end
        end

        % Add best feature to selected features and remove it from remaining features
        selectedFeatures = [selectedFeatures bestFeature];
        remainingFeatures = setdiff(remainingFeatures, bestFeature);
    end

%%
train_data = train_data(:,selectedFeatures);
test_data = test_data(:,selectedFeatures);
%%

svm_model = fitcsvm(train_data, train_labels);
    svm_predicted_labels = predict(svm_model, test_data);
    acc(rep,1,iROI) = mean(svm_predicted_labels == test_labels);

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
    acc(rep,2,iROI) = mean(knn_predicted_labels == test_labels);

end

% Report average decoding accuracies
fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(acc(:,1,iROI))*100);
fprintf('Average Minkowski KNN decoding accuracy: %.2f%%\n', mean(acc(:,2,iROI))*100);

end
toc