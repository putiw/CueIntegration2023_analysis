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

    %% load ROIs
    roi = {'V1','V3','V3A','MT','MST','V7','V4t','FST'};
    roimask = cell(1,numel(roi));
    for i = 1:numel(roi)
        [tmpl, ~, ~] = cvnroimask(sub,'lh','Glasser2016',roi{i},[],[]);
        [tmpr,~,~] = cvnroimask(sub,'rh','Glasser2016',roi{i},[],[]);
        roimask{i} = [tmpl{:};tmpr{:}];
    end
    roimask{9} = roimask{4}|roimask{5}|roimask{7}|roimask{8};
    
%%    

load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPED_FITHRF_GLMDENOISE_RR.mat', 'R2');




%% nuance regressor

nuisance = [];

for iRun = 1:length(runNum)  
tmpregressor = readtable([myDir.bids '/derivatives/fmriprep/' sub '/' ses '/func/' sub '_' ses '_task-Cue_run-',num2str(iRun),'_desc-confounds_timeseries.csv']);
tmpr=[tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter];
nuisance =[nuisance; tmpr];
end       
        
        %%
        whichRoi = 8;
        R2min = 40;
showtimeseries(datafiles(1:5),dsCon(1:5),roimask{whichRoi}&R2>=R2min,10,[1 2;3 4;5 6;7 8],[-1.5 1.5],[])
showtimeseries(datafiles(6:end),dsCon(6:end),roimask{whichRoi}&R2>=R2min,10,[1 2;3 4;5 6;7 8],[-1.5 1.5],[])
showtimeseries(datafiles,dsCon,roimask{whichRoi}&R2>=R2min,10,[1 2;3 4;5 6;7 8],[-1.5 1.5],[])

for iRun = 1:10
showtimeseries(datafiles(iRun),dsCon(iRun),roimask{whichRoi}&R2>=R2min,10,[1 2;3 4;5 6;7 8],[-1.5 1.5],[])
pause(2)
end

