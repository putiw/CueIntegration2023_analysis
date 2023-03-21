clear all;close all;clc;

% design 2 - 1.5 on 3 off 80 trials 
% design 3 - 1.5 on 3 off 60 trials 

[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct,param] = init_decode_ses2;
%%
% R2min = 30;
% for iRun = 3
% showtimeseries(datafiles(iRun),dsCon(iRun),roimask{9}&R2>=R2min,10,[1 2],[-1.5 1.5],[])
% pause(2)
% end

%%
mymask = double(R2>min(maxk(R2(roimask{9}),501)))&roimask{9};
tmp = datafiles{3}(mymask,:);
dms = dsTrial{3};

figure(1), clf
plot(tmp)
ylabel('Raw fMRI signal');
xlabel('TR');
title('Time series of some vertices');
percentageChange = (tmp./mean(tmp,2)-1)*100;
figure(2), clf
plot(percentageChange')
ylabel('fMRI response (% change in image intensity');
xlabel('Frame');
title('Time series of all voxels in the ROI');
hold on
plot(nanmean(percentageChange),'w-','linewidth',1);
figure(3), clf
imagesc(dms)
%% get beta using glm
nT = size(tmp,2);
tau = 2;
delta = 2;
t = [0:1:30];
tshift = max(t-delta,0)
hrf = (tshift/2).^2 .* exp(-tshift/2) / (2*2);
dmsX = conv2(dms,hrf');
dmsX = dmsX(1:nT,:);
drif = 1:nT;
baseline = ones(nT,1);
tmpregressor = '/Users/pw1246/Desktop/MRI/CueIntegration2023/derivatives/fmriprep/sub-0248/ses-02/func/sub-0248_ses-02_task-Cue_run-3_desc-confounds_timeseries.csv';
tmpregressor = readtable(tmpregressor);
nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];        
model = [dmsX drif' baseline nuisance];
model(isnan(model)) = 0;
budgetBetas = (pinv(model) * percentageChange')';
glmbeta = [budgetBetas(:,1:60)' nonzeros(dsCon{3}(:,1)+dsCon{3}(:,2)*2)];
%% get beta using TR
TR = 3;
tr3beta = zeros(60,sum(mymask));
tr4beta = zeros(60,sum(mymask));
whichTrial = find(sum(dsCon{3},2)==1);
for iTrial = 1:numel(whichTrial)
    tr3beta(iTrial,:) = percentageChange(:,whichTrial(iTrial)+TR-1);
    tr4beta(iTrial,:) = percentageChange(:,whichTrial(iTrial)+TR);
end
tr3beta = [tr3beta nonzeros(dsCon{3}(:,1)+dsCon{3}(:,2)*2)];
tr4beta = [tr4beta nonzeros(dsCon{3}(:,1)+dsCon{3}(:,2)*2)];
%%

beta = zeros(60,500,3);
beta(:,:,1) = glmbeta(:,1:end-1);
beta(:,:,2) = tr3beta(:,1:end-1);
beta(:,:,3) = tr3beta(:,1:end-1);
lb = nonzeros(dsCon{3}(:,1)+dsCon{3}(:,2)*2);

nRep = 20;
trials = 1:60;

%%
result = zeros(nRep,3);
for iRep = 1:nRep
    
    trainI = randperm(60,56);
    testI = trials(~ismember(trials,trainI));
    trainLb = lb(trainI);
    testLb = lb(testI);
    
    for ii = 1:3
        
      train = beta(trainI,:,ii);
      
      diff = abs(nanmean(train(trainLb==1,:))-nanmean(train(trainLb==2,:)))./mean([std(train(trainLb==1,:)) std(train(trainLb==2,:))]);      
      train = train(:,diff>min(maxk(diff,51)));
      test = beta(testI,diff>min(maxk(diff,51)),ii);
      
      [trainedClassifier, validationAccuracy] = subknnTrain([train trainLb]);
      fit = trainedClassifier.predictFcn(test);
      result(iRep,ii) = sum((fit == testLb))/numel(testLb);
           
    end
    
end

mean(result)


%%

[coeff,score,latent] = pca(beta(:,:,3));

figure(3);clf
hold on
scatter3(score(lb==1,1),score(lb==1,2),score(lb==1,3),'filled');
scatter3(score(lb==2,1),score(lb==2,2),score(lb==2,3),'filled');
xlabel('pc1');ylabel('pc2');zlabel('pc3');




