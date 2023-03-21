function [datafiles,nuisance,roimask,R2,label,dsCon,dsTrial,dataDim,param] = init_decode(sub)

%% setup path
addpath(genpath(fullfile(pwd, 'funcs')));
param.bids = '/Users/pw1246/Desktop/MRI/CueIntegration2023';
param.git = '/Users/pw1246/Documents/GitHub';
param.drop = '/Users/pw1246/Dropbox (RVL)/2022 Human 3D Motion Cue Isolation/expFiles';
param.proj = 'CueIntegration2023';
addpath(genpath(fullfile(param.git, 'wpToolbox')));
setup_user('puti',param.proj,param.bids,param.git);
tic
param.sub = sub;%'sub-0248';
param.ses = 'ses-01';
hemi = {'L','R'};
param.task = 'cue';

%% set param

runNum = 1:10;
trNum = 360;
trDur = 1;
stimDur = 1;
dsCon = cell(numel(runNum),1);
datafiles = cell(numel(runNum),1);
dsTrial = cell(numel(runNum),1);
design = cell(numel(runNum),1);
dsConAll = [];
hrf = getcanonicalhrf(stimDur,trDur);
dataDim = valstruct_create(param.sub);
baseline = 100;
ind = 1:9:360;

%% load data (gii to mgh)
for iRun = 1:numel(runNum)
    for iH = 1:numel(hemi)
        input = sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-%s_run-%s_space-fsnative_hemi-%s_bold.func.gii',param.bids,param.sub,param.ses,param.sub,param.ses,param.task,num2str(iRun),hemi{iH});
        output = sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-%s_run-%s_space-fsnative_hemi-%s_bold.func.mgh',param.bids,param.sub,param.ses,param.sub,param.ses,param.task,num2str(iRun),hemi{iH});
        if ~exist(output)
            system(['mri_convert ' input ' ' output]);
        end
    end
end
datafiles = load_data(param.bids,param.task,'fsnative','.mgh',param.sub,param.ses,1:10);
%%
nuisance = cell(numel(runNum),1);
for iRun = 1:numel(runNum)
    
    f1 = dir(sprintf('%s/%s/%s/*run-%s_*',param.drop,param.sub,param.ses,num2str(iRun)));
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
        nonactiveVoxels = baseline * ones(trNum,dataDim.numlh);
        activeVoxels = repmat(fmriSignal(:),[1,dataDim.numrh]);
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
    nuisance{iRun} = [tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x str2double(tmpregressor.trans_x_derivative1) tmpregressor.trans_x_power2 tmpregressor.trans_y str2double(tmpregressor.trans_y_derivative1) tmpregressor.trans_y_power2 tmpregressor.trans_z str2double(tmpregressor.trans_z_derivative1) tmpregressor.trans_z_power2 tmpregressor.rot_x str2double(tmpregressor.rot_x_derivative1) tmpregressor.rot_x_power2 tmpregressor.rot_y str2double(tmpregressor.rot_y_derivative1) tmpregressor.rot_y_power2 tmpregressor.rot_z str2double(tmpregressor.rot_z_derivative1) tmpregressor.rot_z_power2 tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];  
    
end
label = cat(1,design{:});
label(label(:,5)==5,:)=[];
label(:,end+1) = repelem(1:size(datafiles,2),size(label,1)/size(datafiles,2))';
%% load ROIs
param.roi = {'V1','V3','V3A','V7','V4t','MT','MST','FST'};
roimask = cell(1,numel(param.roi));
for i = 1:numel(param.roi)
    [tmpl, ~, ~] = cvnroimask(param.sub,'lh','Glasser2016',param.roi{i},[],[]);
    [tmpr,~,~] = cvnroimask(param.sub,'rh','Glasser2016',param.roi{i},[],[]);
    roimask{i} = [tmpl{:};tmpr{:}];
end

%%
R2 = [];
%load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPED_FITHRF_GLMDENOISE_RR.mat', 'R2');

%load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPEC_FITHRF_GLMDENOISE.mat', 'R2');

end