%% knob options
% 'cutoffFreq' - 
% 'whichTR' - double vector - 1:9 etc
% 'znorm' - 1 or 0 or [0 1]
% 'nuisance' - 1 or 0 or [0 1]
% 'nTestTrials' - double scalar 
% 'MRMR' - 1 or 0 or [0 1]
    % 'k' -
% 'wAvg' - 1 or 0 or [0 1]
    % 'nVol' -
% 'whichDecoder' - 'svm'

clear all;close all;clc
addpath(genpath('funcs'));
%[datafiles,param] = init_decode('sub-0201','ses-01');
[datafiles,param] = Copy_of_init_decode('sub-0248','ses-02');
whichRoi = {'V1'};
whichCon = 'monoL';
%% 'whichTR'

whichKnob = 'whichTR';
whatRange = {6:9,7:9};
whatRange = 1:9;
tickVal = string(1:numel(whatRange));
myTitle = {'Which TR do we use?'}; 
 
 
figTR = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

%% 'cutoffFreq'

whichKnob = 'cutoffFreq';
whatRange = 1./(5:5:100);
tickVal = string(5:5:100);
% whatRange = 1./(5:2:100);
% tickVal = string(5:2:100);
myTitle = {'High-pass frequency cutoff'}; 
 
 
figTR = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

%% 'nTestTrials'

whichKnob = 'nTestTrials';
whatRange = 1:25;
tickVal = string(whatRange);
myTitle = {'# of trials in testing data'};
 
 
figNtrial = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

%% 'whichDecoder'

whichKnob = 'whichDecoder';
whatRange = {'svm','knn','naiveBayes','tree','ensemble'};
tickVal = whatRange;                
myTitle = {'Which decoder to use?'};                
 
 

figDecoder = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

%% 'nuisance'

whichKnob = 'nuisance';
whatRange = [0 1];
tickVal ={'no','yes'};
myTitle = {'Regress out nuisance regressors or not'};
 
 
figNuisance = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

%% 'znorm'

whichKnob = 'znorm';
whatRange = [0 1];
tickVal ={'no','yes'};
myTitle = {'Z-normalize or not'};
 
 
figZnorm = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

%% 'MRMR'

whichKnob = 'MRMR';
whatRange = [0 1];
tickVal ={'no','yes'};
myTitle = {'Use MRMR or not'};
 
 
figZnorm = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

%% 'wAvg'

whichKnob = 'wAvg';
whatRange = [0 1];
tickVal ={'no','yes'};
myTitle = {'Use wAvg or not'};
 
 
figZnorm = test_knobs(whichKnob,whatRange,whichRoi,whichCon,datafiles,param,myTitle,tickVal);

