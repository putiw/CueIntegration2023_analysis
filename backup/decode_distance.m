clear all;clc;close all;
[datafiles,roimask,R2,label,dsCon,dsTrial,valstruct] = init_decode;
%%

whichRoi = 9;
whichhemi = [ones(valstruct.numlh,1); ones(valstruct.numrh,1)];
R2min =min(maxk(R2(roimask{whichRoi}&whichhemi),25));
validationAccuracy = zeros(10,4);
close all;
figure;
hold on
for iRun = 1:10
    
   showtimeseries(datafiles(iRun),dsCon(iRun),roimask{whichRoi}&R2>=R2min&whichhemi,10,[1 2;3 4;5 6;7 8],[-1.5 1.5],[])
end

showtimeseries(datafiles,dsCon,roimask{whichRoi}&R2>=R2min,10,[1 2;3 4;5 6;7 8],[-1.5 1.5],[])

   for iRun = 1:10
    beta = getTRdiff(datafiles(iRun),dsCon(iRun),roimask{whichRoi}&R2>=R2min,4);
    for iCon = 1:4
        tmp = [cat(1,beta{iCon*2-1:iCon*2,1}) cat(1,beta{iCon*2-1:iCon*2,2}) cat(1,beta{iCon*2-1:iCon*2,3})];
        tmp(:,end+1) = repelem([1;2],10,1);
                
        %[trainedClassifier, validationAccuracy(iRun,iCon)] = splitcenter(tmp);
        
    end
    
scatter((0.85:3.85)+0.03*iRun,validationAccuracy(iRun,:),50,'MarkerEdgeColor',[0.3 0.3 0.3],'MarkerFaceColor',[1 1 1],'Linewidth',2);
drawnow
end

for iCon = 1:4
line([iCon iCon],[mean(validationAccuracy(:,iCon))+std(validationAccuracy(:,iCon))./sqrt(10)...,
    mean(validationAccuracy(:,iCon))-std(validationAccuracy(:,iCon))./sqrt(10)],'Color',[1 0 0],'Linewidth',2) 
end
scatter(1:4,mean(validationAccuracy),80,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 1 1],'Linewidth',2);


plot([0 5],[0.5 0.5],'k--','Linewidth',1);
ylim([0 1])
xlim([0.5 4.5])
