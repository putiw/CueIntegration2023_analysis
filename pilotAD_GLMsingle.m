%% pilot data at AD

%% setup path
%% load data (or simulate data) and design matrix
%% glm (basic & kay)
%% decode 
%% visualize 


%% setup path
clear all;clc;close all;
myDir.bids = '/Users/pw1246/Desktop/MRI/CueIntegration2023';
myDir.git = '/Users/pw1246/Documents/GitHub';
myDir.drop = '/Users/pw1246/Dropbox (RVL)/2022 Human 3D Motion Cue Isolation/expFiles';
myDir.proj = 'CueIntegration2023';
addpath(genpath(fullfile(myDir.git, 'wpToolbox')));
setup_user('puti',myDir.proj,myDir.bids,myDir.git);
tic
sub = 'sub-0248';
ses = 'ses-01';
hemi = {'L','R'};

%% set param

runNum = 1:10;
trNum = 280;
dsCon = cell(numel(runNum),1);
datafiles = cell(numel(runNum),1);
dsTrial = cell(numel(runNum),1);
design = cell(numel(runNum),1);
dsConAll = [];
hrf = getcanonicalhrf(1.5,1.5);
valstruct = valstruct_create(sub);
baseline = 100;
ind = 1:3:270;

%% load data (gii to mgh)
for iRun = 1:numel(runNum)
    for iH = 1:numel(hemi)
        input = sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-Cue_run-%s_space-fsnative_hemi-%s_bold.func.gii',myDir.bids,sub,ses,sub,ses,num2str(iRun),hemi{iH});
        output = sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-Cue_run-%s_space-fsnative_hemi-%s_bold.func.mgh',myDir.bids,sub,ses,sub,ses,num2str(iRun),hemi{iH});
        if ~exist(output)
        system(['mri_convert ' input ' ' output]);
        end
    end
end
datafiles = load_data(myDir.bids,'Cue','fsnative','.mgh',sub,ses,1:10);
%%
for iRun = 1:numel(runNum)
    
    f1 = dir(sprintf('%s/%s/%s/*run-%s_*',myDir.drop,sub,ses,num2str(iRun)));
    f1 = load([f1.folder '/' f1.name]);
    dsCon{iRun} = f1.pa.dsCon;
    dsTrial{iRun} = f1.pa.dsTrial;
    design{iRun} = f1.pa.design;
    dsConAll = [dsConAll;f1.pa.dsCon];
    
    if isempty(datafiles) % simulate data if no data file
        neuralActivity = zeros(trNum,1);
        neuralActivity(ind) = (f1.pa.design(:,3)-1)*10+f1.pa.design(:,5);
        neuralActivity(neuralActivity==5|neuralActivity==10)=0;
        fmriSignal = baseline + conv(neuralActivity,hrf);
        fmriSignal = fmriSignal(1:length(neuralActivity));
        nonactiveVoxels = baseline * ones(trNum,valstruct.numlh);
        activeVoxels = repmat(fmriSignal(:),[1,valstruct.numrh]);
        data = [activeVoxels nonactiveVoxels];
        noiseSD = 0.1;
        driftRate = 0.02;
        noise = noiseSD * randn(size(data));
        datafiles{iRun} = data + noise + repmat(driftRate * (1:trNum)',1,size(data,2));
    end
end
label = cat(1,design{:});
label(label(:,5)==5,:)=[];
clear data activeVoxels nonactiveVoxels noise driftRate noiseSD fmriSignal neuralActivity f1 input output iH iRun 

%% GLMSingle

if ~exist([pwd '/GLMSingle'],'dir')
    mkdir([pwd '/GLMSingle']);
    opt = struct('wantmemoryoutputs',[1 1 1 1]);
    [results] = GLMestimatesingletrial(dsCon,datafiles,1.5,1.5,[pwd '/GLMSingle'],opt);
    clear models;
    models.B = results{2};
    models.C = results{3};
    models.D = results{4};
    betas = reshape(models.D.modelmd,size(models.D.modelmd,1),size(models.D.modelmd,4));
else
    clear models;
    models.B = load([pwd '/GLMSingle/TYPEB_FITHRF.mat']);
    models.C = load([pwd '/GLMSingle/TYPEC_FITHRF_GLMDENOISE.mat']);
    models.D = load([pwd '/GLMSingle/TYPED_FITHRF_GLMDENOISE_RR.mat']);
end
%% visualize R2
close all
visualizeGLM(sub,models.D.R2,1,[0 100],[],[],colormap('hot'));

%% visualize betas
close all
for iCue = 1:4
figure(iCue)
vals = nanmean(betas(:,label(:,3)==2&label(:,5)==iCue),2);
filter = models.D.R2<=prctile(models.D.R2,90);
%filter = filter | (vals<=prctile(vals,84));
%visualizeGLM(sub,vals,1,[prctile(vals,84) prctile(vals,99.7)],[],filter,colormap('hot'));
%(cmapsign4)
filter = filter | (vals<=0.8303);
visualizeGLM(sub,vals,1,[0.8303 4.7648],[],filter,colormap('hot'));
end

%% decode using GLMsingle betas
    %% load ROIs
    roi = {'V1','V3','V3A','MT','MST','V7','V4t','FST'};
    roimask = cell(1,numel(roi));
    for i = 1:numel(roi)
        [tmpl, ~, ~] = cvnroimask(sub,'lh','Glasser2016',roi{i},[],[]);
        [tmpr,~,~] = cvnroimask(sub,'rh','Glasser2016',roi{i},[],[]);
        roimask{i} = [tmpl{:};tmpr{:}];
    end
    roimask{9} = roimask{4}|roimask{5}|roimask{7}|roimask{8};
    %% prep data
    whichRoi = 8; %5 FST and MST
    iCue = 4; % comb
    data1 = betas(roimask{whichRoi},label(:,5)==iCue);
    label1 = label(label(:,5)==iCue,3);
    




%% glm - across run
dataAll = cat(1,datafiles{:})';
dmsX1 = conv2(blkdiag(dsTrial{:}),hrf');
dmsX1 = dmsX1(1:trNum*10,:);
drif = repmat(1:trNum,1,10);
baseline = ones(trNum*10,1);
model1 = [dmsX1 drif' baseline];
percentageChange1 = (dataAll./mean(dataAll,2)-1)*100;
model1(isnan(model1)) = 0;
budgetBetas1 = (pinv(model1) * percentageChange1')';

%% get roi
roi = {'V1','V3','V3A','MT','MST','V7','V4t','FST'};
roimask = cell(1,numel(roi));
for i = 1:numel(roi)
[tmpl, ~, ~] = cvnroimask(sub,'lh','Glasser2016',roi{i},[],[]);
[tmpr,~,~] = cvnroimask(sub,'rh','Glasser2016',roi{i},[],[]);
roimask{i} = [tmpl{:};tmpr{:}];
end
roimask{9} = roimask{4}|roimask{5}|roimask{7}|roimask{8};

%% decode
nRep = 100;
label = cat(1,design{:});
label(label(:,5)==5,:)=[];
results = zeros(nRep,numel(roimask));
for whichRoi = 4
    tmpData = [budgetBetas1(roimask{whichRoi},1:size(dmsX1,2))'];

    for irep = 1:nRep
        whichTest = randperm(size(dmsX1,2),80);
        whichTrain = find(~ismember(1:size(dmsX1,2),whichTest));
        decode = classify(tmpData(whichTest,:),tmpData(whichTrain,:),label(whichTrain,3),'diaglinear');
        results(irep,whichRoi) = mean(decode==label(whichTest,3));
    end
end
mean(results)
%%


%% glm - per run

%%
ds.con = cell(1,3);
ds.label = [];
run = 1:3;
for iRun = 1:3
    irun = run(iRun);
    onset = 1:3:186;
    tmp = dir(sprintf('%s/expFiles/%s_%s_task-cue_run-%s_*',pwd,sub,ses,num2str(irun)));
    load([tmp.folder '/' tmp.name])
    ds.con{iRun} = zeros(200,2);
    ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==1),1) = 1;
    ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==2),2) = 1;
    ds.label = [ds.label; pa.design(pa.design(:,5)~=2,3)]; 
end


% roi
roi = {'V1','V3','V3A','MT','MST','V7','V4t','FST','TPOJ2'};
roimask = cell(1,numel(roi));
for i = 1:numel(roi)
[tmpl, ~, ~] = cvnroimask('fsaverage','lh','Glasser2016',roi{i},[],[]);
[tmpr,~,~] = cvnroimask('fsaverage','rh','Glasser2016',roi{i},[],[]);
roimask{i} = [tmpl{:};tmpr{:}];
end
roimask{10} = roimask{4}|roimask{5}|roimask{7}|roimask{8};


% run separately
nRep = 200;
accRun = zeros(nRep,10,3);
runAcc = cell(1,3);
runSe = cell(1,3);
for iRun = 1:3
    for whichRoi = 1:10 %FST
        
        label = ds.label(iRun*50-49:iRun*50);
        drif = [1:200];
        baseline = ones(200,1);
        tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(iRun),'_desc-confounds_timeseries.csv']);
        nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
        model1 = [eye(200) baseline drif' nuisance];
        percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
        model1(isnan(model1)) = 0;
        budgetBetas1 = (pinv(model1) * percentageChange1')';
        
        data = normalize(budgetBetas1(:,1:200),2);
        onset = find(sum(ds.con{iRun},2));
        
        dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
        for iO = 1:numel(onset)
            dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
        end
        
        %
        whichData = dataTR65'; %dataTR65';
        
        trial = 1:numel(label);
        
        Class = [find(label==1) find(label==2)];
        
        testN = 2;
        tmin = 90;
        rglz = 1;
        for iRep = 1:nRep
            [whichRoi iRep mean(accRun(:,:,iRun))]

            testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
            
            t1 = whichData(~ismember(trial,testID),:);
            c1 = label(~ismember(trial,testID));
            
            % feature selection
            dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
            [a,crit]=maxk(dp,50);
            
            train = [t1(:,crit) c1];
 
            [trainedClassifier, validationAccuracy] =svmgaus([t1(:,crit) c1],rglz);

            
            t2 = whichData(testID,crit);
            c2 = label(testID);
            
            yfit = trainedClassifier.predictFcn(t2);
            accRun(iRep,whichRoi,iRun) = mean(yfit==c2);
        end
        
    end
    
    runAcc{iRun} = mean(accRun(:,:,iRun));
    runSe{iRun} = std(accRun(:,:,iRun))./sqrt(nRep);
   
end




% 3 runs
nRep = 500;
acc = zeros(nRep,10);

for whichRoi = 1:10 %FST
    
    label = ds.label;
    drif = [1:200];
    baseline = ones(200,1);
    
    whichData = [];
    for iRun = 1:3 %
        tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(iRun),'_desc-confounds_timeseries.csv']);
        nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
        model1 = [eye(200) baseline drif' nuisance ];
        percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
        model1(isnan(model1)) = 0;
        budgetBetas1 = (pinv(model1) * percentageChange1')';
        
        data = normalize(budgetBetas1(:,1:200),2);
        onset = find(sum(ds.con{iRun},2));

         dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
        for iO = 1:numel(onset)
            dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
        end
       
        whichData = [whichData; dataTR65'];
    end
    
    
    trial = 1:numel(label);
    
    Class = [find(label==1) find(label==2)];
    
    testN = 2;
    tmin = 90;
    rglz = 1;
    for iRep = 1:nRep
        [whichRoi iRep mean(acc)]
        %train data, train label, test label
        
        testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
        
        t1 = whichData(~ismember(trial,testID),:);
        c1 = label(~ismember(trial,testID));
        
        % feature selection
        dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
        [a,crit]=maxk(dp,50);
        
        train = [t1(:,crit) c1];
        
        [trainedClassifier, validationAccuracy] =svmgaus(train,rglz);
        
        t2 = whichData(testID,crit);
        c2 = label(testID);
        
        yfit = trainedClassifier.predictFcn(t2);
        %     yfit = classify(t2,t1(:,crit),c1,'diaglinear');
        acc(iRep,whichRoi) = mean(yfit==c2);
    end
    mean(acc)
end
% 
% f = fullfile(pwd,['/decodingAcc/' sub,'-mono-L-run123.mat']);
% meanAcc = mean(acc);
% seAcc = std(acc)./sqrt(nRep);
% save(f,'roi','meanAcc','seAcc','runAcc','runSe');
% %
% 
% datafiles = load_data(bidsDir,'TASK','fsaverage','.mgh',sub,ses,4:6);
% %
% ds.con = cell(1,3);
% ds.label = [];
% run = 4:6;
% for iRun = 1:3
%     irun = run(iRun);
%     onset = 1:3:186;
%     tmp = dir(sprintf('%s/expFiles/%s_%s_task-cue_run-%s_*',pwd,sub,ses,num2str(irun)));
%     load([tmp.folder '/' tmp.name])
%     ds.con{iRun} = zeros(200,2);
%     ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==1),1) = 1;
%     ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==2),2) = 1;
%     ds.label = [ds.label; pa.design(pa.design(:,5)~=2,3)]; 
% end
% 
% % run separately
% nRep = 500;
% accRun = zeros(nRep,10,3);
% runAcc = cell(1,3);
% runSe = cell(1,3);
% for iRun = 1:3
%     for whichRoi = 1:10 %FST
%         
%         label = ds.label(iRun*50-49:iRun*50);
%         drif = [1:200];
%         baseline = ones(200,1);
%         tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(3+iRun),'_desc-confounds_timeseries.csv']);
%         nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
%         model1 = [eye(200) baseline drif' nuisance];
%         percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
%         model1(isnan(model1)) = 0;
%         budgetBetas1 = (pinv(model1) * percentageChange1')';
%         
%         data = normalize(budgetBetas1(:,1:200),2);
%         onset = find(sum(ds.con{iRun},2));
%         
%         dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
%         for iO = 1:numel(onset)
%             dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
%         end
%         
%         %
%         whichData = dataTR65'; %dataTR65';
%         
%         trial = 1:numel(label);
%         
%         Class = [find(label==1) find(label==2)];
%         
%         testN = 2;
%         tmin = 90;
%         rglz = 1;
%         for iRep = 1:nRep
%             [whichRoi iRep mean(accRun(:,:,iRun))]
% 
%             testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
%             
%             t1 = whichData(~ismember(trial,testID),:);
%             c1 = label(~ismember(trial,testID));
%             
%             % feature selection
%             dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
%             [a,crit]=maxk(dp,50);
%             
%             train = [t1(:,crit) c1];
%  
%             [trainedClassifier, validationAccuracy] =svmgaus([t1(:,crit) c1],rglz);
% 
%             
%             t2 = whichData(testID,crit);
%             c2 = label(testID);
%             
%             yfit = trainedClassifier.predictFcn(t2);
%             accRun(iRep,whichRoi,iRun) = mean(yfit==c2);
%         end
%         
%     end
%     
%     runAcc{iRun} = mean(accRun(:,:,iRun));
%     runSe{iRun} = std(accRun(:,:,iRun))./sqrt(nRep);
%    
% end
% 
% % 3 runs
% nRep = 500;
% acc = zeros(nRep,10);
% 
% for whichRoi = 1:10 %FST
%     
%     label = ds.label;
%     drif = [1:200];
%     baseline = ones(200,1);
%     
%     whichData = [];
%     for iRun = 1:3 %
%         tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(3+iRun),'_desc-confounds_timeseries.csv']);
%         nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
%         model1 = [eye(200) baseline drif' nuisance ];
%         percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
%         model1(isnan(model1)) = 0;
%         budgetBetas1 = (pinv(model1) * percentageChange1')';
%         
%         data = normalize(budgetBetas1(:,1:200),2);
%         onset = find(sum(ds.con{iRun},2));
% 
%          dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
%         for iO = 1:numel(onset)
%             dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
%         end
%        
%         whichData = [whichData; dataTR65'];
%     end
%     
%     
%     trial = 1:numel(label);
%     
%     Class = [find(label==1) find(label==2)];
%     
%     testN = 2;
%     tmin = 90;
%     rglz = 1;
%     for iRep = 1:nRep
%         [whichRoi iRep mean(acc)]
%         %train data, train label, test label
%         
%         testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
%         
%         t1 = whichData(~ismember(trial,testID),:);
%         c1 = label(~ismember(trial,testID));
%         
%         % feature selection
%         dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
%         [a,crit]=maxk(dp,50);
%         
%         train = [t1(:,crit) c1];
%         
%         [trainedClassifier, validationAccuracy] =svmgaus(train,rglz);
%         
%         t2 = whichData(testID,crit);
%         c2 = label(testID);
%         
%         yfit = trainedClassifier.predictFcn(t2);
%         %     yfit = classify(t2,t1(:,crit),c1,'diaglinear');
%         acc(iRep,whichRoi) = mean(yfit==c2);
%     end
%     mean(acc)
% end
% 
% f = fullfile(pwd,['/decodingAcc/' sub,'-mono-R-run123.mat']);
% meanAcc = mean(acc);
% seAcc = std(acc)./sqrt(nRep);
% save(f,'roi','meanAcc','seAcc','runAcc','runSe');
% 
% 
% datafiles = load_data(bidsDir,'TASK','fsaverage','.mgh',sub,ses,7:9);
% 
% ds.con = cell(1,3);
% ds.label = [];
% run = 7:9;
% for iRun = 1:3
%     irun = run(iRun);
%     onset = 1:3:186;
%     tmp = dir(sprintf('%s/expFiles/%s_%s_task-cue_run-%s_*',pwd,sub,ses,num2str(irun)));
%     load([tmp.folder '/' tmp.name])
%     ds.con{iRun} = zeros(200,2);
%     ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==1),1) = 1;
%     ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==2),2) = 1;
%     ds.label = [ds.label; pa.design(pa.design(:,5)~=2,3)]; 
% end
% 
% 
% % run separately
% nRep = 500;
% accRun = zeros(nRep,10,3);
% runAcc = cell(1,3);
% runSe = cell(1,3);
% for iRun = 1:3
%     for whichRoi = 1:10 %FST
%         
%         label = ds.label(iRun*50-49:iRun*50);
%         drif = [1:200];
%         baseline = ones(200,1);
%         tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(6+iRun),'_desc-confounds_timeseries.csv']);
%         nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
%         model1 = [eye(200) baseline drif' nuisance];
%         percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
%         model1(isnan(model1)) = 0;
%         budgetBetas1 = (pinv(model1) * percentageChange1')';
%         
%         data = normalize(budgetBetas1(:,1:200),2);
%         onset = find(sum(ds.con{iRun},2));
%         
%         dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
%         for iO = 1:numel(onset)
%             dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
%         end
%         
%         %
%         whichData = dataTR65'; %dataTR65';
%         
%         trial = 1:numel(label);
%         
%         Class = [find(label==1) find(label==2)];
%         
%         testN = 2;
%         tmin = 90;
%         rglz = 1;
%         for iRep = 1:nRep
%             [whichRoi iRep mean(accRun(:,:,iRun))]
% 
%             testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
%             
%             t1 = whichData(~ismember(trial,testID),:);
%             c1 = label(~ismember(trial,testID));
%             
%             % feature selection
%             dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
%             [a,crit]=maxk(dp,50);
%             
%             train = [t1(:,crit) c1];
%  
%             [trainedClassifier, validationAccuracy] =svmgaus([t1(:,crit) c1],rglz);
% 
%             
%             t2 = whichData(testID,crit);
%             c2 = label(testID);
%             
%             yfit = trainedClassifier.predictFcn(t2);
%             accRun(iRep,whichRoi,iRun) = mean(yfit==c2);
%         end
%         
%     end
%     
%     runAcc{iRun} = mean(accRun(:,:,iRun));
%     runSe{iRun} = std(accRun(:,:,iRun))./sqrt(nRep);
%    
% end
% 
% % 3 runs
% nRep = 500;
% acc = zeros(nRep,10);
% 
% for whichRoi = 1:10 %FST
%     
%     label = ds.label;
%     drif = [1:200];
%     baseline = ones(200,1);
%     
%     whichData = [];
%     for iRun = 1:3 %
%         tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(6+iRun),'_desc-confounds_timeseries.csv']);
%         nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
%         model1 = [eye(200) baseline drif' nuisance ];
%         percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
%         model1(isnan(model1)) = 0;
%         budgetBetas1 = (pinv(model1) * percentageChange1')';
%         
%         data = normalize(budgetBetas1(:,1:200),2);
%         onset = find(sum(ds.con{iRun},2));
% 
%          dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
%         for iO = 1:numel(onset)
%             dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
%         end
%        
%         whichData = [whichData; dataTR65'];
%     end
%     
%     
%     trial = 1:numel(label);
%     
%     Class = [find(label==1) find(label==2)];
%     
%     testN = 2;
%     tmin = 90;
%     rglz = 1;
%     for iRep = 1:nRep
%         [whichRoi iRep mean(acc)]
%         %train data, train label, test label
%         
%         testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
%         
%         t1 = whichData(~ismember(trial,testID),:);
%         c1 = label(~ismember(trial,testID));
%         
%         % feature selection
%         dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
%         [a,crit]=maxk(dp,50);
%         
%         train = [t1(:,crit) c1];
%         
%         [trainedClassifier, validationAccuracy] =svmgaus(train,rglz);
%         
%         t2 = whichData(testID,crit);
%         c2 = label(testID);
%         
%         yfit = trainedClassifier.predictFcn(t2);
%         %     yfit = classify(t2,t1(:,crit),c1,'diaglinear');
%         acc(iRep,whichRoi) = mean(yfit==c2);
%     end
%     mean(acc)
% end
% %
% f = fullfile(pwd,['/decodingAcc/' sub,'-bino-run123.mat']);
% meanAcc = mean(acc);
% seAcc = std(acc)./sqrt(nRep);
% save(f,'roi','meanAcc','seAcc','runAcc','runSe');
% %
% datafiles = load_data(bidsDir,'TASK','fsaverage','.mgh',sub,ses,10:12);
% %
% ds.con = cell(1,3);
% ds.label = [];
% run = 10:12;
% for iRun = 1:3
%     irun = run(iRun);
%     onset = 1:3:186;
%     tmp = dir(sprintf('%s/expFiles/%s_%s_task-cue_run-%s_*',pwd,sub,ses,num2str(irun)));
%     load([tmp.folder '/' tmp.name])
%     ds.con{iRun} = zeros(200,2);
%     ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==1),1) = 1;
%     ds.con{iRun}(onset(pa.design(:,5)==1&pa.design(:,3)==2),2) = 1;
%     ds.label = [ds.label; pa.design(pa.design(:,5)~=2,3)]; 
% end
% 
% % run separately
% nRep = 500;
% accRun = zeros(nRep,10,3);
% runAcc = cell(1,3);
% runSe = cell(1,3);
% for iRun = 1:3
%     for whichRoi = 1:10 %FST
%         
%         label = ds.label(iRun*50-49:iRun*50);
%         drif = [1:200];
%         baseline = ones(200,1);
%         tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(9+iRun),'_desc-confounds_timeseries.csv']);
%         nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
%         model1 = [eye(200) baseline drif' nuisance];
%         percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
%         model1(isnan(model1)) = 0;
%         budgetBetas1 = (pinv(model1) * percentageChange1')';
%         
%         data = normalize(budgetBetas1(:,1:200),2);
%         onset = find(sum(ds.con{iRun},2));
%         
%         dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
%         for iO = 1:numel(onset)
%             dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
%         end
%         
%     
%         
%         %
%         whichData = dataTR65'; %dataTR65';
%         
%         trial = 1:numel(label);
%         
%         Class = [find(label==1) find(label==2)];
%         
%         testN = 2;
%         tmin = 90;
%         rglz = 1;
%         for iRep = 1:nRep
%             [whichRoi iRep mean(accRun(:,:,iRun))]
% 
%             testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
%             
%             t1 = whichData(~ismember(trial,testID),:);
%             c1 = label(~ismember(trial,testID));
%             
%             % feature selection
%             dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
%             [a,crit]=maxk(dp,50);
%             
%             train = [t1(:,crit) c1];
%  
%             [trainedClassifier, validationAccuracy] =svmgaus([t1(:,crit) c1],rglz);
% 
%             
%             t2 = whichData(testID,crit);
%             c2 = label(testID);
%             
%             yfit = trainedClassifier.predictFcn(t2);
%             accRun(iRep,whichRoi,iRun) = mean(yfit==c2);
%         end
%         
%     end
%     
%     runAcc{iRun} = mean(accRun(:,:,iRun));
%     runSe{iRun} = std(accRun(:,:,iRun))./sqrt(nRep);
%    
% end
% 
% 
% 
% 
% % 3 runs
% nRep = 500;
% acc = zeros(nRep,10);
% 
% for whichRoi = 1:10 %FST
%     
%     label = ds.label;
%     drif = [1:200];
%     baseline = ones(200,1);
%     
%     whichData = [];
%     for iRun = 1:3 %
%         tmpregressor = readtable(['/Users/pw1246/Desktop/MRI/CueIntegration/derivatives/fmriprep/sub-0801/ses-01/func/sub-0801_ses-01_task-TASK_run-',num2str(9+iRun),'_desc-confounds_timeseries.csv']);
%         nuisance=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
%         model1 = [eye(200) baseline drif' nuisance ];
%         percentageChange1 = datafiles{iRun}(roimask{whichRoi},:);
%         model1(isnan(model1)) = 0;
%         budgetBetas1 = (pinv(model1) * percentageChange1')';
%         
%         data = normalize(budgetBetas1(:,1:200),2);
%         onset = find(sum(ds.con{iRun},2));
% 
%          dataTR65 = zeros(sum(roimask{whichRoi}),numel(onset));
%         for iO = 1:numel(onset)
%             dataTR65(:,iO) = [mean(data(:,onset(iO)+5),2)];
%         end
%         
%        
%         whichData = [whichData; dataTR65'];
%     end
%     
%     
%     trial = 1:numel(label);
%     
%     Class = [find(label==1) find(label==2)];
%     
%     testN = 2;
%     tmin = 90;
%     rglz = 1;
%     for iRep = 1:nRep
%         [whichRoi iRep mean(acc)]
%         %train data, train label, test label
%         
%         testID = [Class(randperm(size(Class,1),testN),1); Class(randperm(size(Class,1),testN),2)];
%         
%         t1 = whichData(~ismember(trial,testID),:);
%         c1 = label(~ismember(trial,testID));
%         
%         % feature selection
%         dp = abs(mean(t1(c1==1,:))-mean(t1(c1==2,:)))./(std(t1(c1==1,:))+std(t1(c1==2,:)));
%         [a,crit]=maxk(dp,50);
%         
%         train = [t1(:,crit) c1];
%         
%         [trainedClassifier, validationAccuracy] =svmgaus(train,rglz);
%         
%         t2 = whichData(testID,crit);
%         c2 = label(testID);
%         
%         yfit = trainedClassifier.predictFcn(t2);
%         %     yfit = classify(t2,t1(:,crit),c1,'diaglinear');
%         acc(iRep,whichRoi) = mean(yfit==c2);
%     end
%     mean(acc)
% end
% %
% f = fullfile(pwd,['/decodingAcc/' sub,'-comb-run123.mat']);
% meanAcc = mean(acc);
% seAcc = std(acc)./sqrt(nRep);
% save(f,'roi','meanAcc','seAcc','runAcc','runSe');
% toc