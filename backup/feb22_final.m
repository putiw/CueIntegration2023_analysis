% final pilot 3 feb 22

clear all;close all;clc;
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct,param] = init_decode_ses3;
% Define the sampling rate (TR) and data
TR = 1.5;
% Define number of repetitions

tic
%%
% Specify the number of features to select using mRMR
k = 400;
% Define number of testing trials
num_test_trials = 2;
num_reps = 100;
acc = zeros(num_reps,4,9);
nVol = 200;
for iROI = 1:9
    data = cellfun(@(m)m(roimask{iROI},:), datafiles, 'UniformOutput', 0);
    data = cat(2,data{:})';
    %% Compute the power spectrum
 
%     [P,f] = pwelch(data,[],[],[],1/TR);
%     
%     % Plot the power spectrum
%     semilogy(f,P);
%     xlabel('Frequency (Hz)');
%     ylabel('Power');
    
    %% High-pass frequency
    % Define the cutoff frequency (in Hz)
    cutoff_freq = 0.025;  %0.025
    
    % Define the filter parameters
    nyquist_freq = 1/(2*TR);
    filter_order = 2;
    Wn = cutoff_freq/nyquist_freq;
    
    % Create the high-pass filter
    [b,a] = butter(filter_order,Wn,'high');
    
    % Apply the filter to the data
    filtered_data = filtfilt(b,a,data);
    %% Regress out noise
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
    

    %%
    dsT = cat(1,dsTrial{:});
    %dms = dsTrial{3};
    dmS = sum(dsT,2);
   
%         tmp = zeros(176,size(regressed_data,2),2);
%     for iT = 4:5
%         tmp(:,:,iT-3)= zscore(regressed_data(find(dmS)-1+iT,:));
%     end
%     tmp=mean(tmp,3);
    
    
    tmp = zeros(176,size(regressed_data,2),4);
    for iT = 4:7
        tmp(:,:,iT-3)= zscore(regressed_data(find(dmS)-1+iT,:));
    end
    tmp=mean(tmp,3);

    %%
    lb = label(:,3);
    %% Define data dimensions
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
        trainData = [class1_data(train_indices, :); class2_data(train_indices2, :)];
        trainLabels = [ones(num_trials-num_test_trials,1); 2*ones(num_trials-num_test_trials,1)];
        testData = [class1_data(test_indices, :); class2_data(test_indices2, :)];
        testLabels = [ones(num_test_trials,1); 2*ones(num_test_trials,1)];
        
        
        % Compute the relevance scores 
        % (correlation between each feature and the labels)
        relevance = zeros(size(trainData, 2), 1);
        for i = 1:size(trainData, 2)
            relevance(i) = abs(corr(trainData(:, i), trainLabels));
        end
        
        % Compute the redundancy scores 
        % (average correlation between each feature and
        % the top k-1 most relevant features)
        redundancy = mean(abs(corr(trainData)))';       
        
        % Compute the mRMR scores (relevance minus redundancy)
        MRMR = relevance - redundancy;
        
        % Select the k features with the highest mRMR scores
        [~, whichFeatures] = sort(MRMR, 'descend');
        whichFeatures = whichFeatures(1:k);
        
        % Subset the data to include only the selected features        
        trainData =  trainData(:,whichFeatures);
        testData =  testData(:,whichFeatures);
        
        %%
                
        model = [(trainLabels-mean(trainLabels)).*2]; % trainLabels>-1
        weight = model \ trainData;
        
        away = find(weight(1,:)>0); % positive weights are away voxels
        w1 = (weight(1,away)').^2;
        [~,idx] = maxk(w1,nVol);
        idx = setdiff(1:size(away,2),idx);
        w1(idx) = 0;
        
        toward = find(weight(1,:)<0); % negative weights are toward voxels
        w2 = abs(weight(1,toward)').^2;
        [~,idx] = maxk(w2,nVol);
        idx = setdiff(1:size(toward,2),idx);
        w2(idx) = 0;
        
        trainData =  [trainData(:,away)*w1/sum(w1) trainData(:,toward)*w2/sum(w2)];
        testData =  [testData(:,away)*w1/sum(w1) testData(:,toward)*w2/sum(w2)];
        
        %%
        % Train and test SVM classifier
        svm_model = fitcsvm(trainData, trainLabels);
        svm_predicted_labels = predict(svm_model, testData);
        acc(rep,1,iROI) = mean(svm_predicted_labels == testLabels);
 
        svm_model = fitcsvm(...
            trainData, ...
            trainLabels, ...
            'KernelFunction', 'gaussian', ...
            'PolynomialOrder', [], ...
            'KernelScale', 22, ...
            'BoxConstraint', 1, ...
            'Standardize', true, ...
            'ClassNames', [1; 2]);
        
        svm_predicted_labels = predict(svm_model, testData);
        
        acc(rep,2,iROI) = mean(svm_predicted_labels == testLabels);
     
classificationKNN = fitcknn(...
    trainData, ...
    trainLabels, ...
    'Distance', 'Minkowski', ...
    'Exponent', 3, ...
    'NumNeighbors', 88, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', false, ...
    'ClassNames', [1; 2]);

        
        svm_predicted_labels = predict(classificationKNN, testData);
        
        acc(rep,3,iROI) = mean(svm_predicted_labels == testLabels);
        
               classificationNaiveBayes = fitcnb(...
        trainData, ...
        trainLabels, ...
        'Kernel', 'Box', ...
        'Support', 'Unbounded', ...
        'DistributionNames', repmat({'Kernel'}, 1, 2), ...
        'ClassNames', [1; 2]);
    
     svm_predicted_labels = predict(classificationNaiveBayes, testData);
        
        acc(rep,4,iROI) = mean(svm_predicted_labels == testLabels);
        
        reshape(mean(acc(1:rep,:,:),1),4,9)
    end
    
    % Report average decoding accuracies
    fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(acc(:,1,iROI))*100);
    fprintf('Average gaussian SVM decoding accuracy: %.2f%%\n', mean(acc(:,2,iROI))*100);
    
end
reshape(mean(acc),4,9)