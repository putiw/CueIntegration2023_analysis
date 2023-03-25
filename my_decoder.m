function [fig, result] = my_decoder(datafiles,param,opt)

acc = zeros(opt.nReps,4,numel(param.roi));
for iCon = 1:4 % loop through different cue conditions
    for iRoi = 1:numel(param.roi)
        
        tmp = zeros(size(param.label,1)/4,sum(param.roimask{iRoi}));
        
        for iRun = 1:numel(datafiles)
            
            data = datafiles{iRun}(param.roimask{iRoi},:)';
            
            if opt.highpass == 1
            % Define the filter parameters
            nyquist_freq = 1/(2*param.trDur);
            filter_order = 2;
            Wn = opt.cutoffFreq/nyquist_freq;
            
            % Create the high-pass filter
            [b,a] = butter(filter_order,Wn,'high');
            
            % Apply the filter to the data
            data = filtfilt(b,a,data);                
            end
            
            % Regress out noise
            tmpNuisance = [];
            if opt.nuisance == 1
                tmpNuisance = param.nuisance{iRun};
            end
            
            % calculate the mean time series (global signal)
            global_signal = mean(data, 2);
            
            % create the design matrix for regression
            X = [ones(size(global_signal)), global_signal,tmpNuisance];
            X(isnan(X))=0;
            % perform regression
            coefficients = X \ data;
            
            % calculate the residuals
            regressed_data = data - X * coefficients;
            %
            
            dmS = sum(param.dsCon{iRun}(:,[iCon*2-1 iCon*2]),2);
            tmpp = zeros(sum(dmS),size(regressed_data,2),numel(opt.whichTR));
            cn = 1;
            for iT = opt.whichTR
                if opt.znorm ==1
                    tmpp(:,:,cn)= zscore(regressed_data(find(dmS)-1+iT,:));
                else
                    tmpp(:,:,cn)= regressed_data(find(dmS)-1+iT,:);
                end
                cn = cn+1;
            end
            
            tmp(iRun*sum(dmS)-sum(dmS)+1:iRun*sum(dmS),:)=mean(tmpp,3); % average z-scores for the seclected TRs
            
        end
        
        currentLabel = param.label(param.label(:,5)==iCon,3);
        
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
            trainLabel = [ones(nTrials-opt.nTestTrials,1); 2*ones(nTrials-opt.nTestTrials,1)];
            testData = [dataClass1(whichTest1, :); dataClass2(whichTest2, :)];
            testLabel = [ones(opt.nTestTrials,1); 2*ones(opt.nTestTrials,1)];
            
            
            if opt.MRMR == 1
                % Compute the relevance scores
                % (correlation between each feature and the labels)
                
                relevance = abs(corr(trainData, trainLabel))';
                
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
                
                
                model = [(trainLabel-mean(trainLabel)).*2]; % trainLabel>-1
                weight = model \ trainData;
                
                %                 w1 = weight';%[~,idx] = maxk(w1,nVol);w1(idx) = 0;
                %
                %                 trainData =  [trainData*w1/sum(w1)];
                %                 testData =  [testData*w1/sum(w1)];
                %
                
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
                
                
                
                trainData =  [trainData(:,away)*w1/sum(w1) trainData(:,toward)*w2/sum(w2)];
                testData =  [testData(:,away)*w1/sum(w1) testData(:,toward)*w2/sum(w2)];
                
                
                
            end
            
            
            % Train and test classifier
            
            switch opt.whichDecoder
                case 'svm'
                    mymodel = fitcsvm(trainData, trainLabel);
                    predictLabels = predict(mymodel, testData);
                    chanceModel = fitcsvm(trainData, trainLabel(randperm(size(trainLabel,1),size(trainLabel,1))));
                    chanceLabels = predict(chanceModel, testData);
                case 'knn'
                    mymodel = fitcknn(trainData,trainLabel);
                    predictLabels = predict(mymodel, testData);
                    chanceModel = fitcknn(trainData, trainLabel(randperm(size(trainLabel,1),size(trainLabel,1))));
                    chanceLabels = predict(chanceModel, testData);
                case 'naiveBayes'
                    mymodel = fitcnb(trainData, trainLabel);
                    predictLabels = predict(mymodel, testData);
                    chanceModel = fitcnb(trainData, trainLabel(randperm(size(trainLabel,1),size(trainLabel,1))));
                    chanceLabels = predict(chanceModel, testData);
                case 'tree'
                    mymodel = fitctree(trainData, trainLabel);
                    predictLabels = predict(mymodel, testData);
                    chanceModel = fitctree(trainData, trainLabel(randperm(size(trainLabel,1),size(trainLabel,1))));
                    chanceLabels = predict(chanceModel, testData);
                case 'ensemble'
                    mymodel = fitcensemble(trainData, trainLabel);
                    predictLabels = predict(mymodel, testData);
                    chanceModel = fitcensemble(trainData, trainLabel(randperm(size(trainLabel,1),size(trainLabel,1))));
                    chanceLabels = predict(chanceModel, testData);
                case 'classify'
                    predictLabels = classify(testData,trainData,trainLabel,'diaglinear');
                    chanceLabels = classify(testData,trainData,trainLabel(randperm(size(trainLabel,1),size(trainLabel,1))),'diaglinear');
            end
            
            acc(rep,iCon,iRoi) = mean(predictLabels == testLabel);
            chanceAcc(rep,iCon,iRoi) = mean(chanceLabels == testLabel);
            squeeze(mean(acc(1:rep,:,:))) % print out results as it goes
            
        end
        
    end
end

toc
%
%close all
barcolor = [251 176 59; 247 147 30; 0 113 188; 0 146 69]./255;
fig = figure('Position', [100, 100, 500, 300]);
hold on
accuracy = squeeze(mean(acc)).*100;
stderr = squeeze(std(acc))./sqrt(opt.nReps-1).*100';
chanceAccuracy = squeeze(mean(chanceAcc)).*100;
chancestderr = squeeze(std(chanceAcc))./sqrt(opt.nReps-1).*100';
chanceBot = chanceAccuracy - chancestderr;
chanceTop = chanceAccuracy + chancestderr;

b1 = bar(1:numel(param.roi),accuracy,'EdgeColor','none');
barW = (b1(1).BarWidth - b1(end).XEndPoints(1) + b1(1).XEndPoints(1))./4;
for ii = 1:numel(b1)
    b1(ii).FaceColor =  barcolor(ii,:);
    b1(ii).FaceAlpha = 0.8;
    line([b1(ii).XEndPoints; b1(ii).XEndPoints],[accuracy(ii,:)+stderr(ii,:);accuracy(ii,:)-stderr(ii,:)],'Color', 'k', 'LineWidth', 1);
    x = [b1(ii).XEndPoints - barW; b1(ii).XEndPoints + barW; b1(ii).XEndPoints + barW; b1(ii).XEndPoints - barW];
    y = [chanceBot(ii,:); chanceBot(ii,:); chanceTop(ii,:); chanceTop(ii,:)]
    patch(x, y, 'k', 'EdgeColor','none','FaceAlpha', 0.2);
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

result.accuracy = accuracy;
result.error = stderr;

result.accuracyChance = accuracy;
result.errorChance = stderr;








end