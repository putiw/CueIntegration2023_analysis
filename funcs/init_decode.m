function [datafiles,param] = init_decode(whichSub,whichSes)

%% setup path

addpath(genpath(fullfile(pwd, 'funcs')));
param.proj = 'CueIntegration2023';
% param.proj = 'pilot';
param.bids = '/Users/pw1246/Desktop/MRI/CueIntegration2023';
param.git = '/Users/pw1246/Documents/GitHub';
param.fsDir = '/Applications/freesurfer/7.2.0';
param.drop = '/Users/pw1246/Dropbox (RVL)/2022 Human 3D Motion Cue Isolation/expFiles';
% param.drop = '/Users/pw1246/RVL Dropbox/Puti Wen/2022 Human 3D Motion Cue Isolation/expFiles_pilot';
addpath(genpath(fullfile(param.git, 'wpToolbox')));
setup_user(param.proj,param.bids,param.git,param.fsDir);
tic
param.sub = whichSub;
param.ses = whichSes;
hemi = {'L','R'};
param.task = 'cue';

%% set param

runNum = 1:10;
trNum = 360;
param.trDur = 1;
stimDur = 1;
param.dsCon = cell(numel(runNum),1);
datafiles = cell(numel(runNum),1);
param.dsTrial = cell(numel(runNum),1);
design = cell(numel(runNum),1);
dsConAll = [];
hrf = getcanonicalhrf(stimDur,param.trDur);
param.dataDim = valstruct_create(param.sub);
baseline = 100;
TR = 1; % dur of TR in secs

%% load data

datafiles = load_data(param.bids,param.task,'fsnative','.mgh',param.sub,param.ses,runNum);

%% load design matrix and param.nuisance regressors

param.nuisance = cell(numel(runNum),1);
for iRun = 1:numel(runNum)
    
    f1 = dir(sprintf('%s/%s/%s/*run-%s_*',param.drop,param.sub,param.ses,num2str(iRun)));
    f1 = load([f1.folder '/' f1.name]);
    param.dsCon{iRun} = f1.pa.dsCon;
    param.dsTrial{iRun} = f1.pa.dsTrial;
    design{iRun} = f1.pa.design;
    
    if isempty(datafiles) % simulate data if no data file
        ind = 1:(f1.pa.trialDuration+f1.pa.ITI)./param.TR:(trNum-f1.pa.pause/param.TR);
        
        neuralActivity = zeros(trNum,1);
        neuralActivity(ind) = (f1.pa.design(:,3)-1)*10+f1.pa.design(:,5);
        neuralActivity(neuralActivity==5|neuralActivity==10)=0;
        fmriSignal = baseline + conv(neuralActivity,hrf);
        fmriSignal = fmriSignal(1:length(neuralActivity));
        nonactiveVoxels = baseline * ones(trNum,param.dataDim.numlh);
        activeVoxels = repmat(fmriSignal(:),[1,param.dataDim.numrh]);
        data = [activeVoxels nonactiveVoxels];
        noiseSD = 0.1;
        driftRate = 0.02;
        noise = noiseSD * randn(size(data));
        datafiles{iRun} = data + noise + repmat(driftRate * (1:trNum)',1,size(data,2));
    end
    
    tmpregressor = [param.bids '/derivatives/fmriprep/' param.sub '/' param.ses '/func/' param.sub '_' param.ses '_task-' param.task '_run-' num2str(iRun) '_desc-confounds_timeseries'];
    if ~exist([tmpregressor '.csv'],'file')
        system([mv [tmpregressor '.tsv'] [tmpregressor '.csv']])
    end
    tmpregressor = readtable([tmpregressor '.csv']);
    param.nuisance{iRun} = [tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x str2double(tmpregressor.trans_x_derivative1) tmpregressor.trans_x_power2 tmpregressor.trans_y str2double(tmpregressor.trans_y_derivative1) tmpregressor.trans_y_power2 tmpregressor.trans_z str2double(tmpregressor.trans_z_derivative1) tmpregressor.trans_z_power2 tmpregressor.rot_x str2double(tmpregressor.rot_x_derivative1) tmpregressor.rot_x_power2 tmpregressor.rot_y str2double(tmpregressor.rot_y_derivative1) tmpregressor.rot_y_power2 tmpregressor.rot_z str2double(tmpregressor.rot_z_derivative1) tmpregressor.rot_z_power2 tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];
    
end
param.label = cat(1,design{:});
param.label(param.label(:,5)==5,:)=[];
param.label(:,end+1) = repelem(1:size(datafiles,2),size(param.label,1)/size(datafiles,2))';

%% load ROIs

param.roi = {'V1','V3','V3A','V7','V4t','MT','MST','FST'};
param.roimask = get_roi(whichSub,'Glasser2016',param.roi);

%%
param.R2 = [];
%load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPED_FITHRF_GLMDENOISE_RR.mat', 'param.R2');
%load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPEC_FITHRF_GLMDENOISE.mat', 'param.R2');

end