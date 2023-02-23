clear all;close all;clc;

% design 1 - block, 10.5 on 10.5 off 20 trials 
% design 2 - 1.5 on 3 off 80 trials 
% design 3 - 1.5 on 3 off 60 trials 

[datafiles,roimask,~,label,dsCon,dsTrial,valstruct,param] = init_decode_ses2;

%% GLM 1 

myresults = GLMestimatemodel(dsCon{1}(85:end,:),datafiles{1}(:,85:end),10.5,1.5,'assume',[],0);

close all
figure(1);clf
bins = 1:1:100;
datatoplot = myresults.R2 ;
cmap0 = cmaplookup(bins,min(bins),max(bins),[],hot);

datatoplot(isnan(datatoplot)) = 0;
datatoplot(myresults.R2<=10) = 0;
[rawimg,Lookup,rgbimg] = cvnlookup(param.sub,1,datatoplot,[min(bins) max(bins)],cmap0,0,[],0,{'roiname',{'Glasser2016'},'roicolor',{'w'},'drawroinames',0,'roiwidth',{1},'fontsize',20}); %MT_exvivo %Kastner2015

color = [0.5];
[r,c,t] = size(rgbimg);
[i j] = find(all(rgbimg == repmat(color,r,c,3),3));

for ii = 1: length(i)
    rgbimg(i(ii),j(ii),:) = ones(1,3);
end

set(gcf,'Position',[ 277         119        1141         898])
a = imagesc(rgbimg); axis image tight;
axis off
hold on
% subplot(2,1,2)
plot(0,0);
colormap(cmap0);
hcb=colorbar('SouthOutside');
hcb.Ticks = [0:0.25:1];
% hcb.TickLabels = {'}
hcb.FontSize = 25
hcb.Label.String = 'R2%'
hcb.TickLength = 0.001;

title(param.sub)

figure(2); clf
betas = myresults.modelmd{2};

away = nanmean(betas(:,1),2);
toward = nanmean(betas(:,2),2);
 
C = [1 -1]'
contrast = C' * [away toward]';

datatoplot = contrast' .* double(myresults.R2>10);

datatoplot(datatoplot==0) = -50;
datatoplot(isnan(datatoplot)) = -50;

bins = -0.5:0.01:0.5
cmap0 = cmaplookup(bins,min(bins),max(bins),[],(cmapsign4));
[rawimg,Lookup,rgbimg] = cvnlookup(param.sub,1,datatoplot,[min(bins) max(bins)],cmap0,min(bins),[],0,{'roiname',{'Glasser2016'},'roicolor',{'w'},'drawroinames',1,'roiwidth',{1},'fontsize',20});

color = [0.5];
[r,c,t] = size(rgbimg);
[i j] = find(all(rgbimg == repmat(color,r,c,3),3));

for ii = 1: length(i)
    rgbimg(i(ii),j(ii),:) = ones(1,3);
end

a = imagesc(rgbimg); axis image tight;

    
set(gcf,'Position',[ 277         119        1141         898])
axis off
hold on
% subplot(2,1,2)
plot(0,0);
colormap(cmap0);
hcb=colorbar('SouthOutside');
hcb.Ticks = [0 1];
hcb.TickLabels = {'-0.5';'0.5'}
hcb.FontSize = 25
hcb.TickLength = 0.001;

figure(3);clf

mymask = double(myresults.R2>min(maxk(myresults.R2(roimask{9}),101)))&roimask{9};
sum(mymask)

tcs = nanmean(cat(3,datafiles{1}),3);
ObsResp = nanmean(tcs(logical(mymask),:),1);
dc = nanmean(ObsResp)
ObsResp = 100 * (ObsResp - dc) / dc;

plot(ObsResp,'linewidth',2)
% tcs
hold on

stem(dsCon{1}(:,1),'MarkerFaceColo','red'); %away
stem(dsCon{1}(:,2),'MarkerFaceColo','green'); %toward

xlim([85 280])
legend box off
ylabel('%BOLD')
xlabel('TRs')
set(gca,'Fontsize',15)

%%

%% GLM 1

tmp = datafiles{1}(roimask{9},85:end);
dms = dsCon{1}(85:end,:);

nTrial = 14;
betas = zeros(sum(roimask{9}),nTrial);
ind = 1:14:14*14;
for ii = 1:nTrial
    betas(:,ii)=nanmean(tmp(:,ind(ii)+3:ind(ii)+9),2);
end


aa = [betas' nonzeros(dms(:,1)+dms(:,2)*2)];

%% PCA

[coeff,score,latent] = pca(aa(:,1:end-1),'NumComponents',6);
%%
aaa = [score nonzeros(dms(:,1)+dms(:,2)*2)];