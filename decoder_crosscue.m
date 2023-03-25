function [fig, result] = decoder_crosscue(datafiles,param,opt)

acc = zeros(opt.nReps,4,numel(param.roi));
    for iRoi = 1:numel(param.roi)
        
        tmp = zeros(size(param.label,1)/4,sum(param.roimask{iRoi}),4);
        
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
            
            for iCon = 1:4
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
                
                tmp(iRun*sum(dmS)-sum(dmS)+1:iRun*sum(dmS),:,iCon)=mean(tmpp,3); % average z-scores for the seclected TRs
            end
        end
        
        
        monoL = tmp(:,1);
        monoLlb = param.label(param.label(:,5)==1,3);
        monoR = tmp(:,2);
        monoRlb = param.label(param.label(:,5)==2,3);
        bino = tmp(:,3);
        binolb = param.label(param.label(:,5)==3,3);
        comb = tmp(:,4);
        comblb = param.label(param.label(:,5)==4,3);
        
        mymodel = fitcsvm(monoL, monoLlb);
        predictLabels = predict(mymodel, monoR);
        monoL2R = mean(predictLabels == monoRlb);
        

        mymodel = fitcsvm(comb, comblb);
        
        predictLabels = predict(mymodel, monoL);
        comb2L = mean(predictLabels == monoLlb);
        
        predictLabels = predict(mymodel, monoR);
        comb2R = mean(predictLabels == monoRlb);
        
        predictLabels = predict(mymodel, bino);
        comb2B = mean(predictLabels == binolb);
        
        
%         chanceModel = fitcsvm(trainData, trainLabel(randperm(size(trainLabel,1),size(trainLabel,1))));
%         chanceLabels = predict(chanceModel, testData);               
%         chanceAcc(iCon,iRoi) = mean(chanceLabels == testLabel);
        
        
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