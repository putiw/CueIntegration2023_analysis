% 1. load data
% 2. initialize paramters
% 3. high-pass filter
% 4. regressout global signals and nuisance motion regressors
% 5. z-normalize within each nth TR after onsets and then average
% 6. feature selection/extraction - mRMR and weighted average
% 7. binary decode using SVM/KNN
% 8. visualize


%% load data, nuisance regressors, design matrices, rois, size, etc

clear all;close all;clc;
[datafiles,nuisance,roimask,R2,label,dsCon,dsTrial,dataDim,param] = init_decode('sub-0248');
TR = 1; % dur of TR in secs

%% Set the knobs

opt.whichTR = 6:9; % which TR after onset to use for decode
opt.cutoffFreq = 0.025;  % Define the cutoff high-pass frequency (in Hz)
opt.nTestTrials = 2; % define number of testing trials per class
opt.nReps = 200;% define number of repetitions
opt.MRMR = 0; % use MRMR or not
opt.k = 450; % specify the number of features to select using mRMR
opt.wAvg = 0; % use weighted average or not
opt.nVol = 100; % number of top important features to use for weighted average

%% decode
tic
acc = zeros(opt.nReps,4,numel(param.roi));
for iCon = 1:4 % loop through different cue conditions 
    for iRoi = 1:numel(param.roi)
        
        tmp = zeros(size(label,1)/4,sum(roimask{iRoi}));
        
        for iRun = 1:10
            
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
            nuisance = [];
            tmpregressor = [param.bids '/derivatives/fmriprep/' param.sub '/' param.ses '/func/' param.sub '_' param.ses '_task-' param.task '_run-' num2str(iRun) '_desc-confounds_timeseries.csv'];
            tmpregressor = readtable(tmpregressor);
            tmpN = [tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x str2double(tmpregressor.trans_x_derivative1) tmpregressor.trans_x_power2 tmpregressor.trans_y str2double(tmpregressor.trans_y_derivative1) tmpregressor.trans_y_power2 tmpregressor.trans_z str2double(tmpregressor.trans_z_derivative1) tmpregressor.trans_z_power2 tmpregressor.rot_x str2double(tmpregressor.rot_x_derivative1) tmpregressor.rot_x_power2 tmpregressor.rot_y str2double(tmpregressor.rot_y_derivative1) tmpregressor.rot_y_power2 tmpregressor.rot_z str2double(tmpregressor.rot_z_derivative1) tmpregressor.rot_z_power2 tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];
            opt.nuisance = [nuisance;tmpN];
            
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
                trainData =  trainData(:,whichFeatures(1:opt.k));
                testData =  testData(:,whichFeatures(1:opt.k));
            end
            
            if opt.wAvg == 1
                
                
                model = [(trainLabels-mean(trainLabels)).*2]; % trainLabels>-1
                weight = model \ trainData;
                %
                %         away = find(weight(1,:)>0); % positive weights are away voxels
                %         w1 = (weight(1,away)').^2;
                %         [~,idx] = maxk(w1,nVol);
                %         idx = setdiff(1:size(away,2),idx);
                %         w1(idx) = 0;
                %
                %         toward = find(weight(1,:)<0); % negative weights are toward voxels
                %         w2 = abs(weight(1,toward)').^2;
                %         [~,idx] = maxk(w2,nVol);
                %         idx = setdiff(1:size(toward,2),idx);
                %         w2(idx) = 0;
                %
                
                w1 = weight';%[~,idx] = maxk(w1,nVol);w1(idx) = 0;
                
                
                trainData =  [trainData*w1/sum(w1)];
                testData =  [testData*w1/sum(w1)];
            end
            %
            %         trainData =  [trainData(:,away)*w1/sum(w1) trainData(:,toward)*w2/sum(w2)];
            %         testData =  [testData(:,away)*w1/sum(w1) testData(:,toward)*w2/sum(w2)];
            %
            
            
            %%
            %         figure(iCon);clf;hold on;
            %         scatter(trainData(trainLabels==1,1),trainData(trainLabels==1,2),'r','filled');
            %         scatter(trainData(trainLabels==2,1),trainData(trainLabels==2,2),'b','filled');
            %         scatter(testData(testLabels==1,1),testData(testLabels==1,2),'r','filled');
            %         scatter(testData(testLabels==2,1),testData(testLabels==2,2),'b','filled');
            %         %%
            
            % Train and test SVM classifier
            svm_model = fitcsvm(trainData, trainLabels);
            svm_predicted_labels = predict(svm_model, testData);
            acc(rep,iCon,iRoi) = mean(svm_predicted_labels == testLabels);
            
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
            %         svm_predicted_labels = predict(svm_model, testData);
            %
            %         acc(rep,2,iRoi) = mean(svm_predicted_labels == testLabels);
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
            %         svm_predicted_labels = predict(classificationKNN, testData);
            %
            %         acc(rep,iRoi) = mean(svm_predicted_labels == testLabels);
            %
            
            %         classificationNaiveBayes = fitcnb(...
            %         trainData, ...
            %         trainLabels, ...
            %         'Kernel', 'Box', ...
            %         'Support', 'Unbounded', ...
            %         'DistributionNames', repmat({'Kernel'}, 1, 2), ...
            %         'ClassNames', [1; 2]);
            %
            %      svm_predicted_labels = predict(classificationNaiveBayes, testData);
            %
            %         acc(rep,4,iRoi) = mean(svm_predicted_labels == testLabels);
        end
        
        % Report average decoding accuracies
        %     fprintf('Average SVM decoding accuracy: %.2f%%\n', mean(acc(:,iRoi))*100);
        %     fprintf('Average gaussian SVM decoding accuracy: %.2f%%\n', mean(acc(:,2,iRoi))*100);
        %
    end
end

toc
%%
close all
f2 = figure('Position', [100, 100, 500, 250]);
hold on
accuracy = squeeze(mean(acc)).*100,squeeze(std(acc))';
stderr = squeeze(std(acc))./sqrt(opt.nReps-1).*100';
b1 = bar(1:numel(param.roi),accuracy,'EdgeColor','none');
for ii = 1:numel(b1)
    line([b1(ii).XEndPoints; b1(ii).XEndPoints],[accuracy(ii,:)+stderr(ii,:);accuracy(ii,:)-stderr(ii,:)],'Color', 'k', 'LineWidth', 1);
end
plot([0.4 numel(param.roi)+0.6],[50 50],'k--','LineWidth',1)
xticks(1:numel(param.roi))
xticklabels(param.roi);
ylim([30 80]);
xlim([0.4 numel(param.roi)+0.6])
ylabel('Decoding accuracy (%)');
xlabel('ROIs');
title('Trial-wise decoding accuracy by different cues')
legend('monoL','monoR','bino','comb','Orientation','horizontal');
set(gca,'FontSize',15)
set(gca,'TickDir','out')
box off


