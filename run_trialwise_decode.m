% 1. load data
% 2. initialize paramters
% 3. high-pass filter
% 4. regressout global signals and param.nuisance motion regressors
% 5. z-normalize within each nth TR after onsets and then average
% 6. feature selection/extraction - mRMR and weighted average
% 7. binary decode using SVM/KNN
% 8. visualize

%% load data, param.nuisance regressors, design matrices, rois, size, etc

clear all;close all;clc;
[datafiles,param] = init_decode('sub-0248','ses-02');
%%
%[tmp1,tmp2] = init_decode('sub-0248','ses-02');
%%
% datafiles = [datafiles tmp1];
% param.dsCon = [param.dsCon; tmp2.dsCon];
% param.dsTrial = [param.dsTrial; tmp2.dsTrial];
% param.nuisance = [param.nuisance; tmp2.nuisance];
% param.label = [param.label; tmp2.label];
%     
%% Set the knobs
opt.nReps = 50;% define number of repetitions
opt.whichTR = 6:9; %6:9; % which TR after onset to use for decode
opt.znorm = 1; % z-normalize for each TR before average or not
opt.highpass = 1; % use high-pass filter or not
opt.cutoffFreq = 0.025;  % Define the cutoff high-pass frequency (in Hz)
opt.nuisance = 1; % regress out param.nuisance regressor or not
opt.nTestTrials = 5; % define number of testing trials per class
opt.MRMR = 0; % use MRMR or not
opt.k = 450; % specify the number of features to select using mRMR
opt.wAvg = 0; % use weighted average or not
opt.nVol = 100; % number of top important features to use for weighted average
opt.whichDecoder = 'svm'; % which classifier to use for decode 
   
%% decode
[fig, results] = my_decoder(datafiles,param,opt);

