% try with ses-01 using new method

clear all;clc;close all;
[datafiles,nuisance,roimask,R2,label,dsCon,dsTrial,dataDim,param] = init_decode_backup('sub-0248');

TR = 1.5; % dur of TR in secs

%% Set the knobs
opt.whichTR = 5; %6:9; % which TR after onset to use for decode
opt.cutoffFreq = 0.025;  % Define the cutoff high-pass frequency (in Hz)
opt.nuisance = 1; % regress out nuisance regressor or not
opt.nTestTrials = 1; % define number of testing trials per class
opt.nReps = 50;% define number of repetitions
opt.MRMR = 0; % use MRMR or not
opt.k = 450; % specify the number of features to select using mRMR
opt.wAvg = 0; % use weighted average or not
opt.nVol = 100; % number of top important features to use for weighted average

% decode
tic
acc = zeros(opt.nReps,4,numel(param.roi));

for iCon = 1;%:4 % loop through different cue conditions 
    for iRoi = 8%1:numel(param.roi)
        
        tmp = zeros(size(label,1)/4,sum(roimask{iRoi}));
        
        for iRun = 1:size(datafiles,2)
            
            data = datafiles{iRun}(roimask{iRoi},:)';

            % Define the filter parameters
            nyquist_freq = 1/(2*TR);
            filter_order = 2;
            Wn = opt.cutoffFreq/nyquist_freq;
            
            % Create the high-pass filter
            [b,a] = butter(filter_order,Wn,'high');
            
            % Apply the filter to the data
            filtered_data = filtfilt(b,a,data);
            
            % Regress out noise
            tmpNuisance = [];
            if opt.nuisance == 1    
                tmpNuisance = nuisance{iRun};
            end
            
            % calculate the mean time series (global signal)
            global_signal = mean(filtered_data, 2);
            
            % create the design matrix for regression
            X = [ones(size(global_signal)), global_signal,tmpNuisance];
            X(isnan(X))=0;
            % perform regression
            coefficients = X \ filtered_data;
            
            % calculate the residuals
            regressed_data = filtered_data - X * coefficients;
            %
            
            dmS = sum(dsCon{iRun}(:,[iCon*2-1 iCon*2]),2);
            tmpp = zeros(sum(dmS),size(regressed_data,2),numel(opt.whichTR));
            cn = 1;
            for iT = opt.whichTR
                tmpp(:,:,cn)= zscore(regressed_data(find(dmS)-1+iT,:));
                cn = cn+1;
            end
            
            tmp(iRun*sum(dmS)-sum(dmS)+1:iRun*sum(dmS),:)=mean(tmpp,3); % average z-scores for the seclected TRs
            
            
        end
        
        currentLabel = label(label(:,5)==iCon,3);
        
        dataClass1 = tmp(currentLabel==1,:); % away
        dataClass2 =  tmp(currentLabel==2,:); % toward
        
        % Define data dimensions
        nTrials = size(dataClass1,1); % how many trials do we have for each class
        
        
        for rep = 1:opt.nReps
            
            %         Generate random indices for testing and training data
            whichTest1 = sort(randperm(nTrials, opt.nTestTrials));
            whichTrain1 = setdiff(1:nTrials, whichTest1);
            whichTest2 = sort(randperm(nTrials, opt.nTestTrials));
            whichTrain2 = setdiff(1:nTrials, whichTest2);
            % %
            
            % Separate training and testing data
            trainData = [dataClass1(whichTrain1, :); dataClass2(whichTrain2, :)];
            trainLabels = [ones(nTrials-opt.nTestTrials,1); 2*ones(nTrials-opt.nTestTrials,1)];
            testData = [dataClass1(whichTest1, :); dataClass2(whichTest2, :)];
            testLabels = [ones(opt.nTestTrials,1); 2*ones(opt.nTestTrials,1)];
            
            
            if opt.MRMR == 1
                % Compute the relevance scores
                % (correlation between each feature and the labels)
                
                relevance = corr(trainData, trainLabels)';
                
                % Compute the redundancy scores
                % (average correlation between each feature and
                % the top k-1 most relevant features)
                redundancy = mean(abs(corr(trainData)))';
                
                % Compute the mRMR scores (relevance minus redundancy)
                MRMR = relevance - redundancy;
                
                % Select the k features with the highest mRMR scores
                [~, whichFeatures] = sort(MRMR, 'descend');
                
                % Subset the data to include only the selected features
                k = max(opt.k,size(trainData,2));
                trainData =  trainData(:,whichFeatures(1:k));
                testData =  testData(:,whichFeatures(1:k));
            end
            
            if opt.wAvg == 1
                
                
                model = [(trainLabels-mean(trainLabels)).*2]; % trainLabels>-1
                weight = model \ trainData;
                
                        away = find(weight(1,:)>0); % positive weights are away voxels
                        w1 = (weight(1,away)').^2;
                        [~,idx] = maxk(w1,opt.nVol);
                        idx = setdiff(1:size(away,2),idx);
                        w1(idx) = 0;
                
                        toward = find(weight(1,:)<0); % negative weights are toward voxels
                        w2 = abs(weight(1,toward)').^2;
                        [~,idx] = maxk(w2,opt.nVol);
                        idx = setdiff(1:size(toward,2),idx);
                        w2(idx) = 0;
                
                
%                w1 = weight';%[~,idx] = maxk(w1,nVol);w1(idx) = 0;
%                 
%                 
%                 trainData =  [trainData*w1/sum(w1)];
%                 testData =  [testData*w1/sum(w1)];
        trainData =  [trainData(:,away)*w1/sum(w1) trainData(:,toward)*w2/sum(w2)];
                    testData =  [testData(:,away)*w1/sum(w1) testData(:,toward)*w2/sum(w2)];
            
            end
            %
          
            
            
            %%
            %         figure(iCon);clf;hold on;
            %         scatter(trainData(trainLabels==1,1),trainData(trainLabels==1,2),'r','filled');
            %         scatter(trainData(trainLabels==2,1),trainData(trainLabels==2,2),'b','filled');
            %         scatter(testData(testLabels==1,1),testData(testLabels==1,2),'r','filled');
            %         scatter(testData(testLabels==2,1),testData(testLabels==2,2),'b','filled');
            %         %%
            
            % Train and test SVM classifier
            mymodel = fitcsvm(trainData, trainLabels);
           % svm_model = fitcsvm(trainData, trainLabels(randperm(size(trainLabels,1),size(trainLabels,1))));
            %mymodel = fitcnb(trainData, trainLabels);
            %mymodel = fitcknn(trainData,trainLabels);
            predictLabels = predict(mymodel, testData);
            
            
          %  predictLabels = classify(testData,trainData,trainLabels,'diaglinear');
            
            acc(rep,iCon,iRoi) = mean(predictLabels == testLabels);
            
            squeeze(mean(acc(1:rep,:,:)))
            %
            % svm_model = fitcsvm(...
            %     trainData, ...
            %     trainLabels, ...
            %     'KernelFunction', 'polynomial', ...
            %     'PolynomialOrder', 2, ...
            %     'KernelScale', 1, ...
            %     'BoxConstraint', 948.5057602405659, ...
            %     'Standardize', false, ...
            %     'ClassNames', [1; 2]);
            %
            %
            %         predictLabels = predict(svm_model, testData);
            %
            %         acc(rep,2,iRoi) = mean(predictLabels == testLabels);
            %
            %         classificationKNN = fitcknn(...
            %             trainData, ...
            %             trainLabels, ...
            %             'Distance', 'Minkowski', ...
            %             'Exponent', 3, ...
            %             'NumNeighbors', 88, ...
            %             'DistanceWeight', 'Equal', ...
            %             'Standardize', false, ...
            %             'ClassNames', [1; 2]);
            %
            %
            %         predictLabels = predict(classificationKNN, testData);
            %
            %         acc(rep,iRoi) = mean(predictLabels == testLabels);
            %
            
            %         classificationNaiveBayes = fitcnb(...
            %         trainData, ...
            %         trainLabels, ...
            %         'Kernel', 'Box', ...
            %         'Support', 'Unbounded', ...
            %         'DistributionNames', repmat({'Kernel'}, 1, 2), ...
            %         'ClassNames', [1; 2]);
            %
            %      predictLabels = predict(classificationNaiveBayes, testData);
            %
            %         acc(rep,4,iRoi) = mean(predictLabels == testLabels);
        end
        
        % Report average decoding accuracies
        %     fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(acc(:,iRoi))*100);
        %     fprintf('Average gaussian SVM decoding accuracy: %.2f%%\n', mean(acc(:,2,iRoi))*100);
        %
    end
end

toc
%
close all
barcolor = [251 176 59; 247 147 30; 0 113 188; 0 146 69]./255;
f2 = figure('Position', [100, 100, 500, 250]);
hold on
accuracy = squeeze(mean(acc)).*100,squeeze(std(acc))';
stderr = squeeze(std(acc))./sqrt(opt.nReps-1).*100';
b1 = bar(1:numel(param.roi),accuracy,'EdgeColor','none');
for ii = 1:numel(b1)
    b1(ii).FaceColor =  barcolor(ii,:);
    b1(ii).FaceAlpha = 0.8;
    line([b1(ii).XEndPoints; b1(ii).XEndPoints],[accuracy(ii,:)+stderr(ii,:);accuracy(ii,:)-stderr(ii,:)],'Color', 'k', 'LineWidth', 1);
end
plot([0.4 numel(param.roi)+0.6],[50 50],'k--','LineWidth',1)
xticks(1:numel(param.roi))
xticklabels(param.roi);
ylim([30 100]);
xlim([0.4 numel(param.roi)+0.6])
ylabel('Decoding accuracy (%)');
xlabel('ROIs');
title('Trial-wise decoding accuracy by different cues')
legend('monoL','monoR','bino','comb','Orientation','horizontal');
set(gca,'FontSize',15)
set(gca,'TickDir','out')
box off


%%
featureMethod = 'MRMR'; %'Weight'
% Define the sampling rate (TR) and data
TR = 1.5;
% Define number of repetitions
k = 400;
nVol = 200;
tic

conNow = 2;
% Specify the number of features to select using mRMR
% Define number of testing trials
num_test_trials = 10;
num_reps = 100;

acc = zeros(num_reps,4,9);
for iROI = 1:9
    
    tmp = zeros(size(label,1)/4,sum(roimask{iROI}));
    for iRun = 1:10
        data = datafiles{iRun}(roimask{iROI},:)';
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
        tmpregressor = ['/Users/pw1246/Desktop/MRI/CueIntegration2023/derivatives/fmriprep/sub-0248/ses-01/func/sub-0248_ses-01_task-Cue_run-' num2str(iRun) '_desc-confounds_timeseries.csv'];
        tmpregressor = readtable(tmpregressor);
        tmpN = [tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x str2double(tmpregressor.trans_x_derivative1) tmpregressor.trans_x_power2 tmpregressor.trans_y str2double(tmpregressor.trans_y_derivative1) tmpregressor.trans_y_power2 tmpregressor.trans_z str2double(tmpregressor.trans_z_derivative1) tmpregressor.trans_z_power2 tmpregressor.rot_x str2double(tmpregressor.rot_x_derivative1) tmpregressor.rot_x_power2 tmpregressor.rot_y str2double(tmpregressor.rot_y_derivative1) tmpregressor.rot_y_power2 tmpregressor.rot_z str2double(tmpregressor.rot_z_derivative1) tmpregressor.rot_z_power2 tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];
        nuisance = [nuisance;tmpN];
        
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
        dmS = sum(dsTrial{iRun}(:,find(label(:,5)==conNow&label(:,6)==iRun)-(iRun-1)*80),2);
        
        tmpp = zeros(sum(dmS),size(regressed_data,2),4);
        for iT = 4:7
            tmpp(:,:,iT-3)= zscore(regressed_data(find(dmS)-1+iT,:));
        end
        tmp(iRun*sum(dmS)-sum(dmS)+1:iRun*sum(dmS),:)=mean(tmpp,3);
        
    end
    
    
    
    
    lb = label(label(:,5)==conNow,3);
    
    class1_data = tmp(lb==1,:);
    class2_data =  tmp(lb==2,:);
    
    % Define data dimensions
    num_trials = size(class1_data,1);
    
    
    for rep = 1:num_reps
        
%         Generate random indices for testing and training data
        test_indices = sort(randperm(num_trials, num_test_trials));
        train_indices = setdiff(1:num_trials, test_indices);
        test_indices2 = sort(randperm(num_trials, num_test_trials));
        train_indices2 = setdiff(1:num_trials, test_indices2);
% %         
%         test_indices = 61:70;
%         train_indices = [1:60 71:100];
%          test_indices2 = 61:70;
%         train_indices2 = [1:60 71:100];
        
        % Separate training and testing data
        trainData = [class1_data(train_indices, :); class2_data(train_indices2, :)];
        trainLabels = [ones(num_trials-num_test_trials,1); 2*ones(num_trials-num_test_trials,1)];
        testData = [class1_data(test_indices, :); class2_data(test_indices2, :)];
        testLabels = [ones(num_test_trials,1); 2*ones(num_test_trials,1)];
        
        
        switch featureMethod
            
            case 'Weight'
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
        
        
        
         case 'MRMR'
             
             
                     
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
               
        
        end
        
        %%
        close all;figure(1);clf;hold on;
        scatter(trainData(trainLabels==1,1),trainData(trainLabels==1,2),'r');
        scatter(trainData(trainLabels==2,1),trainData(trainLabels==2,2),'b');
        scatter(testData(testLabels==1,1),testData(testLabels==1,2),'r','filled');
        scatter(testData(testLabels==2,1),testData(testLabels==2,2),'b','filled');
        %%
        
        % Train and test SVM classifier
        svm_model = fitcsvm(trainData, trainLabels);
        svm_predicted_labels = predict(svm_model, testData);
        acc(rep,1,iROI) = mean(svm_predicted_labels == testLabels);
        
%         svm_model = fitcsvm(...
%             trainData, ...
%             trainLabels, ...
%             'KernelFunction', 'gaussian', ...
%             'PolynomialOrder', [], ...
%             'KernelScale', 22, ...
%             'BoxConstraint', 1, ...
%             'Standardize', true, ...
%             'ClassNames', [1; 2]);
        
svm_model = fitcsvm(...
    trainData, ...
    trainLabels, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 1, ...
    'BoxConstraint', 948.5057602405659, ...
    'Standardize', false, ...
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