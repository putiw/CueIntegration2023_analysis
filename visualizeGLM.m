function visualizeGLM(sub,val,view,range,cutoff,filter,cmapc)

clf
bins = min(range):((max(range)-min(range))/20):max(range);
datatoplot = double(val);
cmap0 = cmaplookup(bins,min(bins),max(bins),[],cmapc);
datatoplot(filter) = nan;
valsalpha = ~isnan(datatoplot);
datatoplot(isnan(datatoplot)) = 0;


[rawimg,Lookup,rgbimg] = cvnlookup(sub,view,datatoplot,[min(bins) max(bins)],cmap0,cutoff,[],0,{'overlayalpha',valsalpha,'roiname',{'Glasser2016'},'roicolor',{'g'},'drawroinames',1,'roiwidth',{0.3},'fontsize',5});

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
hcb.Ticks = [0:0.5:1];
hcb.TickLabels = {num2str(min(bins)),num2str(min(bins)+(max(bins)-min(bins))./2),num2str(max(bins))};
hcb.FontSize = 25
hcb.Label.String = 'R2%'
hcb.TickLength = 0.001;
end