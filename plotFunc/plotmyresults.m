monol = [0.4913    0.4500    0.5787   0.5600  0.6075 0.5475    0.5800        0.7212 ];
monor = [0.5413    0.4775    0.6613    0.6150    0.5262 0.6587    0.5763       0.6600];
bino = [0.4938    0.5450    0.5350    0.4913    0.6637    0.4662    0.4550    0.5675];
comb = [0.4763    0.5887    0.4925    0.5075    0.6125    0.5188    0.5887    0.5600];

close all;
f1 = figure; 
bar(1:numel(param.roi),[monol;monor;bino;comb].*100)
hold on
plot([0.2 numel(param.roi)+0.8],[50 50],'k--','LineWidth',1)
xticklabels(param.roi);
ylim([30 80]);
xlim([0.2 numel(param.roi)+0.8])
ylabel('Percentage accuracy (%)');
xlabel('ROIs');
legend('monoL','monoR','bino','comb','Orientation','horizontal');
set(gca,'FontSize',15)
set(gca,'TickDir','out')
box off

%%
close all;
%%
myAcc =[
    0.5162    0.5312    0.5363    0.5625    0.4988    0.5850    0.5363    0.5000
    0.4263    0.4238    0.4487    0.5288    0.5212    0.6300    0.6100    0.5400
    0.5650    0.5075    0.5962    0.5587    0.4825    0.4387    0.5513    0.4888
    0.6400    0.5487    0.4988    0.4062    0.5200    0.5337    0.6412    0.3975];

myAcc2 = [
    0.4863    0.5563    0.5162    0.5225    0.5025    0.4888    0.5563    0.5112
    0.4238    0.3887    0.5075    0.5300    0.5125    0.6250    0.5713    0.5425
    0.5387    0.4875    0.5737    0.4788    0.4213    0.4412    0.4900    0.5363
    0.6038    0.5238    0.4763    0.3663    0.5088    0.5962    0.6138    0.4025];

f2 = figure; 
bar(1:numel(param.roi),[myAcc].*100)
hold on
plot([0.2 numel(param.roi)+0.8],[50 50],'k--','LineWidth',1)
xticklabels(param.roi);
ylim([30 80]);
xlim([0.2 numel(param.roi)+0.8])
ylabel('Percentage accuracy (%)');
xlabel('ROIs');
legend('monoL','monoR','bino','comb','Orientation','horizontal');
set(gca,'FontSize',15)
set(gca,'TickDir','out')
box off