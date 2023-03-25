function myfig = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

cues = {'monoL','monoR','bino','comb'};
opt.nReps = 50;% define number of repetitions
acc = zeros(opt.nReps,numel(whatRange));
chanceAcc = zeros(opt.nReps,numel(whatRange));
iCon = find(strcmp(cues, whichCon)); % set cue conditions
iRoi = find(strcmp(param.roi, whichRoi));

for iRange = 1:numel(whatRange)
    tic
    % default knobs
    opt.whichTR = 6:9; %6:9; % which TR after onset to use for decode
    opt.znorm = 1; % z-normalize for each TR before average or not
    opt.cutoffFreq = 0.025;  % Define the cutoff high-pass frequency (in Hz)
    opt.nuisance = 1; % regress out param.nuisance regressor or not
    opt.nTestTrials = 3; % define number of testing trials per class
    
    opt.MRMR = 0; % use MRMR or not
    opt.k = 450; % specify the number of features to select using mRMR
    opt.wAvg = 0; % use weighted average or not
    opt.nVol = 100; % number of top important features to use for weighted average
    opt.whichDecoder = 'svm'; % which classifier to use for decode
    
    % test this knob
    if iscell(whatRange)
    opt.(char(whichKnob)) = whatRange{iRange};
    else
    opt.(char(whichKnob)) = whatRange(iRange);
    end
    
    tmp = zeros(size(param.label,1)/4,sum(param.roimask{iRoi}));
    
    for iRun = 1:10
        
        data = datafiles{iRun}(param.roimask{iRoi},:)';
        
        % Define the filter parameters
        nyquist_freq = 1/(2*param.trDur);
        filter_order = 2;
        Wn = opt.cutoffFreq/nyquist_freq;
        
        % Create the high-pass filter
        [b,a] = butter(filter_order,Wn,'high');
        
        % Apply the filter to the data
        filtered_data = filtfilt(b,a,data);
        
        % Regress out noise
        tmpNuisance = [];
        if opt.nuisance == 1
            tmpNuisance = param.nuisance{iRun};
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
        
        acc(rep,iRange) = mean(predictLabels == testLabel);
        chanceAcc(rep,iRange) = mean(chanceLabels == testLabel);
        
    end
    

    disp(strjoin([whichKnob ': ' string(whatRange(iRange)) ' -> ' sprintf('%.1f',mean(acc(:,iRange)).*100) '% -> ' sprintf('%.1f', toc) ' secs']))



end

%
barcolor = [251 176 59; 247 147 30; 0 113 188; 0 146 69]./255;

myfig = figure('Position', [100, 100, max(numel(myTitle{:})*15,sum(strlength(tickVal)).*27), 300]);
hold on
accuracy = squeeze(mean(acc)).*100;
stderr = squeeze(std(acc))./sqrt(opt.nReps-1).*100';
b1 = bar(1:numel(whatRange),accuracy,'EdgeColor','none');
b1.FaceAlpha = 0.8;
line([b1.XEndPoints; b1.XEndPoints],[accuracy+stderr;accuracy-stderr],'Color', 'k', 'LineWidth', 1);
b1.FaceColor =  barcolor(iCon,:);
b1.FaceAlpha = 0.8;
plot([0.4 numel(whatRange)+0.6],[50 50],'--','Color',[0.2 0.2 0.2],'LineWidth',1);


chanceAccuracy = squeeze(mean(chanceAcc)).*100;
chancestderr = squeeze(std(chanceAcc))./sqrt(opt.nReps-1).*100';
chanceBot = chanceAccuracy - chancestderr;
chanceTop = chanceAccuracy + chancestderr;
for i = 1:length(b1.XData)
    x = [b1.XData(i) - b1.BarWidth/2, b1.XData(i) + b1.BarWidth/2, b1.XData(i) + b1.BarWidth/2, b1.XData(i) - b1.BarWidth/2];
    y = [chanceBot(i), chanceBot(i), chanceTop(i), chanceTop(i)];
    patch(x, y, 'k', 'EdgeColor','none','FaceAlpha', 0.2);
end


xticks(1:numel(whatRange));

xticklabels(tickVal);

ylim([30 100]);
xlim([0.4 numel(whatRange)+0.6]);
ylabel('Decoding accuracy (%)');
xlabel(whichKnob);
title(myTitle);
newStr = split(myTitle,[" "]);
n = find(cumsum(strlength(newStr)) >= sum(strlength(newStr))./2, 1, 'first');
ax = gca;
if myfig.Position(3) <= sum(strlength(ax.Title.String))*9
    title({strjoin(newStr(1:n)), strjoin(newStr(n+1:end))});
else
end
set(gca,'FontSize',15)
set(gca,'TickDir','out');
box off



end
