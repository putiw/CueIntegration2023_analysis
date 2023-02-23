clear all;clc;close all;
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct] = init_decode;
%%
whichhemi = [ones(valstruct.numlh,1); zeros(valstruct.numrh,1)];
%visualizeGLM('sub-0248',R2,1,[20 100],[],[],colormap('hot'));
roi = {'V1','V3','V3A','MT','MST','V7','V4t','FST','MT+'};
beta = reshape(modelmd,size(modelmd,1),size(modelmd,4));
R2min = 90;
close all;
for iCon = 1:4
    figure(iCon);hold on;
    for iRoi = 5
        tmp = zeros(sum(roimask{iRoi}&R2>=prctile(R2(roimask{iRoi}),R2min)),2);
        for iDir = 1:2
            tmp(:,iDir) = nanmean(beta(roimask{iRoi}&R2>=prctile(R2(roimask{iRoi}),R2min),label(:,5)==iCon&label(:,3)==iDir),2);
        end
        
        plot(tmp(tmp(:,2)>=tmp(:,1),:)','Color',[0 0 1 0.2]);
        plot(nanmean(tmp(tmp(:,2)>=tmp(:,1),:)),'Color',[0 0 1],'linewidth',4);
        plot(tmp(tmp(:,2)<tmp(:,1),:)','Color',[1 0 0 0.2]);
        plot(nanmean(tmp(tmp(:,2)<tmp(:,1),:)),'Color',[1 0 0],'linewidth',4);
        drawnow
        
    end
end
%%
load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPED_FITHRF_GLMDENOISE_RR.mat', 'modelmd','R2');
% load('/Users/pw1246/Documents/GitHub/CueIntegration2023_analysis/GLMSingle/TYPEC_FITHRF_GLMDENOISE.mat', 'modelmd','R2'); 
beta = reshape(modelmd,size(modelmd,1),size(modelmd,4));
%%
close all;
R2min = 90;
colmap = [0 1 0; 0 0 1; 1 0 0; 0 0 0];

for iRoi = [1 3 4 5 7 8]
    tmp = zeros(sum(roimask{iRoi}&R2>=prctile(R2(roimask{iRoi}),R2min)),2,4);
    
    for iCon = 1:4
        for iDir = 1:2
            tmp(:,iDir,iCon) = nanmean(beta(roimask{iRoi}&R2>=prctile(R2(roimask{iRoi}),R2min),label(:,5)==iCon&label(:,3)==iDir),2);
        end
    end
    [~,ind] = maxk(tmp(:,1,1)-tmp(:,2,1),20);
    
    figure(iRoi);hold on;
    for iCon = 1:4
        plot(tmp(ind,:,iCon)','Color',[colmap(iCon,:) 0.2]);
        plot(nanmean(tmp(ind,:,iCon)),'Color',colmap(iCon,:),'linewidth',4);
%         ylim([1 2.5])
        drawnow
    end
end
%% whether monoL and MonoR is same or different
close all

indL = nanmean(beta(:,label(:,5)==1&label(:,3)==1),2)>nanmean(beta(:,label(:,5)==1&label(:,3)==2),2);
indR = nanmean(beta(:,label(:,5)==2&label(:,3)==1),2)>nanmean(beta(:,label(:,5)==2&label(:,3)==2),2);
diff = zeros(size(R2));
diff(indL==indR)=1;
diff((~roimask{9})|R2<prctile(R2(roimask{9}),90))=nan;
visualizeGLM('sub-0248',diff,1,[0 1],[],[],colormap('jet'));  %colormap('jet') cmapsign4





%%
close all;
R2min = 90;
colmap = [0 1 0; 0 0 1; 1 0 0; 0 0 0];

for iRoi = 8
    tmp = zeros(sum(roimask{iRoi}&R2>=prctile(R2(roimask{iRoi}),R2min)),2,4);
    
    for iCon = 4
        for iDir = 1:2
            tmp(:,iDir,iCon) = nanmean(beta(roimask{iRoi}&R2>=prctile(R2(roimask{iRoi}),R2min),label(:,5)==iCon&label(:,3)==iDir),2);
        end
    end
   
end




