clear all;clc;close all;
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct,param] = init_decode;
%% run separately
nRep = 100;
accRun = zeros(nRep,4,size(roimask,2),size(datafiles,2));
runAcc = cell(1,size(datafiles,2));
runSe = cell(1,size(datafiles,2));

for iRun = 1:size(datafiles,2)
    
    for whichRoi = 3% 1:size(roimask,2) %FST
     
        drif = [1:size(datafiles{1},2)];
        baseline = ones(size(datafiles{1},2),1);
        
        tmpregressor = readtable(sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-Cue_run-%s_desc-confounds_timeseries.csv',param.bids,param.sub,param.ses,param.sub,param.ses,num2str(iRun)));
        nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
        model1 = [eye(size(datafiles{1},2)) baseline drif' nuisance];
        percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
        model1(isnan(model1)) = 0;
        budgetBetas1 = (pinv(model1) * percentageChange1')';
        
        data = normalize(budgetBetas1(:,1:size(datafiles{1},2)),2);
        
        onset = find(sum(dsTrial{iRun},2));
        
        whichData = zeros(numel(onset),sum(roimask{whichRoi}));
        for iO = 1:numel(onset)
           % whichData(iO,:) = [data(:,onset(iO)+3)-data(:,onset(iO)+2)];
            whichData(iO,:) = data(:,onset(iO)+3);
        end       
        
        [coeff, score, latent] = pca(whichData,'NumComponents',3);
        close all;hold on;
        ind = find(label(:,6)==iRun&label(:,5)==iCue);  
        scatter3(score(label(ind,3)==1,1), score(label(ind,3)==1,2),score(label(ind,3)==1,3), 'MarkerFaceColor', [1 0 0]);
        scatter3(score(label(ind,3)==2,1), score(label(ind,3)==2,2),score(label(ind,3)==2,3), 'MarkerFaceColor', [0 0 1]);
        
        
        
        for iCue = 4
            
            trial = find(label(:,6)==iRun&label(:,5)==iCue);            
            tmpLabel = label(trial,3);                        
            Class = [trial(label(trial,3)==1) trial(label(trial,3)==2)];            
            testN = 1;
            
            for iRep = 1:nRep 
                iRep
                testID = [Class(randperm(size(trial(label(trial,3)==1),1),testN),1); Class(randperm(size(trial(label(trial,3)==1),1),testN),2)];               
                t1 = whichData(trial(~ismember(trial,testID)),:);
                c1 = tmpLabel(~ismember(trial,testID));
                t2 = whichData(testID,crit);
                c2 = label(testID,3); 
        

                                
                dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
                [a,crit]=maxk(dp,50);                           
                
                [coeff, score, latent] = pca(t1(:,crit),'NumComponents',2);
                proj_data =  t2 * score;


                %[trainedClassifier, validationAccuracy] =trainKNN([t1(:,crit) c1]);   
                [trainedClassifier, validationAccuracy] =sbkn([t1(:,crit) c1]);            
                             
                yfit = trainedClassifier.predictFcn(t2);                
                accRun(iRep,iCue,whichRoi,iRun) = mean(yfit==c2);
            end
        end
        mean(accRun(:,:,whichRoi,iRun))
               
    end   
    
end




%% all runs

whichRoi = 3;
betas = cell(10,1);
drif = [1:size(datafiles{1},2)];
baseline = ones(size(datafiles{1},2),1);

for iRun = 1:size(datafiles,2)
                  
        tmpregressor = readtable(sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-Cue_run-%s_desc-confounds_timeseries.csv',param.bids,param.sub,param.ses,param.sub,param.ses,num2str(iRun)));
        nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
        model1 = [eye(size(datafiles{1},2)) baseline drif' nuisance];
        percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
        model1(isnan(model1)) = 0;
        budgetBetas1 = (pinv(model1) * percentageChange1')';        
        data = normalize(budgetBetas1(:,1:size(datafiles{1},2)),2);       
        onset = find(sum(dsTrial{iRun},2));        
        betas{iRun} = zeros(numel(onset),sum(roimask{whichRoi}));
        for iO = 1:numel(onset)
            betas{iRun}(iO,:) = data(:,onset(iO)+3);
        end 
end
betas = cat(1,betas{:});
        [coeff, score, latent] = pca(betas,'NumComponents',10);
        close all;hold on;
        scatter3(score(label(:,3)==1,1), score(label(:,3)==1,2),score(label(:,3)==1,3), 'MarkerFaceColor', [1 0 0]);
        scatter3(score(label(:,3)==2,1), score(label(:,3)==2,2),score(label(:,3)==2,3), 'MarkerFaceColor', [0 0 1]);
xlabel('pc1');ylabel('pc2');zlabel('pc3')

%% all runs (design matrix try new )

whichRoi = 5;
betas = cell(10,1);
drif = [1:size(datafiles{1},2)];
baseline = ones(size(datafiles{1},2),1);

for iRun = 1:size(datafiles,2)
                  
        tmpregressor = readtable(sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-Cue_run-%s_desc-confounds_timeseries.csv',param.bids,param.sub,param.ses,param.sub,param.ses,num2str(iRun)));
        nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
        model1 = [eye(size(datafiles{1},2)) baseline drif' nuisance];
        percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
        model1(isnan(model1)) = 0;
        budgetBetas1 = (pinv(model1) * percentageChange1')';      
        
       data = normalize(budgetBetas1(:,1:size(datafiles{1},2)),2);       

        onset = find(sum(dsTrial{iRun},2));        
        betas{iRun} = zeros(numel(onset),sum(roimask{whichRoi}));
        for iO = 1:numel(onset)
            betas{iRun}(iO,:) = data(:,onset(iO)+3);
        end 
end
betas = cat(1,betas{:});
    
[coeff, score, latent] = pca(betas,'NumComponents',10);

iCue = 4;
ind1 = label(:,3)==1&label(:,5)==iCue;
ind2 = label(:,3)==2&label(:,5)==iCue;

close all;hold on;
scatter3(score(ind1,1), score(ind1,2),score(ind1,3), 'MarkerFaceColor', [1 0 0]);
scatter3(score(ind2,1), score(ind2,2),score(ind2,3), 'MarkerFaceColor', [0 0 1]);
xlabel('pc1');ylabel('pc2');zlabel('pc3')