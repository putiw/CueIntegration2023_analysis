% try with ses-01 using new method

clear all;clc;close all;
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct] = init_decode;

%%

% Define the sampling rate (TR) and data
TR = 1.5;
% Define number of repetitions

tic
%%
conNow = 4;
% Specify the number of features to select using mRMR
% Define number of testing trials
num_test_trials = 5;
num_reps = 100;
nVol = 300;
acc = zeros(num_reps,3,9);
for iROI = 1:9
    data = cellfun(@(m)m(roimask{iROI},:), datafiles, 'UniformOutput', 0);
    data = cat(2,data{:})';
    % Compute the power spectrum
 
%     [P,f] = pwelch(data,[],[],[],1/TR);
%     
%     % Plot the power spectrum
%     semilogy(f,P);
%     xlabel('Frequency (Hz)');
%     ylabel('Power');
    
    % High-pass frequency
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
    % Regress out noise
    nuisance = [];
    for iRun = 1:10
        tmpregressor = ['/Users/pw1246/Desktop/MRI/CueIntegration2023/derivatives/fmriprep/sub-0248/ses-01/func/sub-0248_ses-01_task-Cue_run-' num2str(iRun) '_desc-confounds_timeseries.csv'];
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
    %
    dsT = cat(1,dsTrial{:});
    dmS = sum(dsT,2);
    
    tmp = zeros(sum(dmS),size(regressed_data,2),3);
    for iT = 4:6
        tmp(:,:,iT-3)= zscore(regressed_data(find(dmS)-1+iT,:));
    end
    tmp=mean(tmp,3);
    
    
    

    lb = label(:,3);
    lbcon = label(:,5);
    
    class1_data = tmp(lb==1&lbcon==conNow,:);
    class2_data =  tmp(lb==2&lbcon==conNow,:);
    
    % Define data dimensions
    num_trials = size(class1_data,1);

    
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
            
        
        model = [trainLabels==1 trainLabels>-1];
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
        
        reshape(mean(acc(1:rep,:,:),1),3,9)
    end
    
    % Report average decoding accuracies
    fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(acc(:,1,iROI))*100);
    fprintf('Average gaussian SVM decoding accuracy: %.2f%%\n', mean(acc(:,2,iROI))*100);
    
end
reshape(mean(acc),3,9)