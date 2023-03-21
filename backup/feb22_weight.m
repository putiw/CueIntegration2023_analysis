% test voxel weights
whichRoi = 8;
data = cellfun(@(m)m(roimask{whichRoi},:), datafiles, 'UniformOutput', 0);
data = cellfun(@(m)100*(m-nanmean(m,2))./nanmean(m,2), data, 'UniformOutput', 0);
data = cat(2,data{:})';
dCon = cat(1,dsCon{:});
dTrial = cat(1,dsTrial{:});
dms = sum(dTrial,2);
cutoff_freq = 0.025;  %0.025
nyquist_freq = 1/(2*TR);
filter_order = 2;
Wn = cutoff_freq/nyquist_freq;
[b,a] = butter(filter_order,Wn,'high');
filtered_data = filtfilt(b,a,data);

nuisance = [];
for iRun = 1:4
    tmpregressor = ['/Users/pw1246/Desktop/MRI/CueIntegration2023/derivatives/fmriprep/sub-0248/ses-03/func/sub-0248_ses-03_task-Cue_run-' num2str(iRun) '_desc-confounds_timeseries.csv'];
    tmpregressor = readtable(tmpregressor);
    tmp = [tmpregressor.global_signal str2double(tmpregressor.global_signal_derivative1) tmpregressor.csf tmpregressor.global_signal_power2 tmpregressor.white_matter tmpregressor.trans_x str2double(tmpregressor.trans_x_derivative1) tmpregressor.trans_x_power2 tmpregressor.trans_y str2double(tmpregressor.trans_y_derivative1) tmpregressor.trans_y_power2 tmpregressor.trans_z str2double(tmpregressor.trans_z_derivative1) tmpregressor.trans_z_power2 tmpregressor.rot_x str2double(tmpregressor.rot_x_derivative1) tmpregressor.rot_x_power2 tmpregressor.rot_y str2double(tmpregressor.rot_y_derivative1) tmpregressor.rot_y_power2 tmpregressor.rot_z str2double(tmpregressor.rot_z_derivative1) tmpregressor.rot_z_power2 tmpregressor.a_comp_cor_00 tmpregressor.t_comp_cor_00];
    nuisance = [nuisance;tmp];
end

global_signal = mean(filtered_data, 2);
X = [ones(size(global_signal)), global_signal,nuisance];
X(isnan(X))=0;
coefficients = X \ filtered_data;
residuals = filtered_data - X * coefficients;
tmp = zeros(176,size(residuals,2),4);
for iT = 4:7
    tmp(:,:,iT-3)= zscore(residuals(find(dms)-1+iT,:));
end
tmp=mean(tmp,3);

model = [label(:,3)==1 label(:,3)>-1];
weight = model \ tmp

nVol = 200;
away = find(weight(1,:)>0);
toward = find(weight(1,:)<0);
x1 = tmp(:,away); %away
w1 = (weight(1,away)').^2;
[~,idx] = maxk(w1,nVol);
idx = setdiff(1:size(away,2),idx);
w1(idx) = 0;
x2 = tmp(:,toward); %toward 
w2 = abs(weight(1,toward)').^2;
[~,idx] = maxk(w2,nVol);
idx = setdiff(1:size(toward,2),idx);
w2(idx) = 0;
aa= [x1*w1/sum(w1) x2*w2/sum(w2) label(:,3)];


