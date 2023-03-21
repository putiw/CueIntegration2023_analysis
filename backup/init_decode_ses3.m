function [datafiles,roimask,R2,label,dsCon,dsTrial,valstruct,param] = init_decode_ses2

%% setup path
addpath(genpath(fullfile(pwd, 'funcs')));
param.bids = '/Users/pw1246/Desktop/MRI/CueIntegration2023';
param.git = '/Users/pw1246/Documents/GitHub';
param.drop = '/Users/pw1246/Dropbox (RVL)/2022 Human 3D Motion Cue Isolation/expFiles';
param.proj = 'CueIntegration2023';
addpath(genpath(fullfile(param.git, 'wpToolbox')));
setup_user('puti',param.proj,param.bids,param.git);
tic
param.sub = 'sub-0248';
param.ses = 'ses-03';
hemi = {'L','R'};

%% set param

runNum = 1:4;
dsCon = cell(numel(runNum),1);
dsTrial = cell(numel(runNum),1);
design = cell(numel(runNum),1);

valstruct = valstruct_create(param.sub);


%% load data (gii to mgh)
for iRun = 1:numel(runNum)
    for iH = 1:numel(hemi)
        input = sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-Cue_run-%s_space-fsnative_hemi-%s_bold.func.gii',param.bids,param.sub,param.ses,param.sub,param.ses,num2str(iRun),hemi{iH});
        output = sprintf('%s/derivatives/fmriprep/%s/%s/func/%s_%s_task-Cue_run-%s_space-fsnative_hemi-%s_bold.func.mgh',param.bids,param.sub,param.ses,param.sub,param.ses,num2str(iRun),hemi{iH});
        if ~exist(output)
        system(['mri_convert ' input ' ' output]);
        end
    end
end
datafiles = load_data(param.bids,'Cue','fsnative','.mgh',param.sub,param.ses,1:4);
%%
for iRun = 1:numel(runNum)
    
    f1 = dir(sprintf('%s/%s/%s/*run-%s_*',param.drop,param.sub,param.ses,num2str(iRun)));
    f1 = load([f1.folder '/' f1.name]);
    dsCon{iRun} = f1.pa.dsCon;
    dsTrial{iRun} = f1.pa.dsTrial;
    design{iRun} = f1.pa.design;    

end
label = cat(1,design{:});
    %% load ROIs
    param.roi = {'V1','V3','V3A','MT','MST','V7','V4t','FST'};
    roimask = cell(1,numel(param.roi));
    for i = 1:numel(param.roi)
        [tmpl, ~, ~] = cvnroimask(param.sub,'lh','Glasser2016',param.roi{i},[],[]);
        [tmpr,~,~] = cvnroimask(param.sub,'rh','Glasser2016',param.roi{i},[],[]);
        roimask{i} = [tmpl{:};tmpr{:}];
    end
    roimask{9} = roimask{4}|roimask{5}|roimask{7}|roimask{8};
    
%%    
R2= [];
load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPED_FITHRF_GLMDENOISE_RR.mat', 'R2');

%load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPEC_FITHRF_GLMDENOISE.mat', 'R2');

end