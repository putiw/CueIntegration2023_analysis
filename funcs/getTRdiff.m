function beta = getTR(datafiles,dsCon,whichRoi,TR)

% given samples and design matrix

beta = cell(size(dsCon{1},2),3);
tmp = cellfun(@(m)m(whichRoi,:), datafiles, 'UniformOutput', 0);
tmp = cellfun(@(m)100*(m-nanmean(m,2))./nanmean(m,2), tmp, 'UniformOutput', 0);
tmp = cat(2,tmp{:})';
label = cat(1,dsCon{:});
beta = cellfun(@(m)zeros(sum(label(:,1)),size(tmp,2)), beta, 'UniformOutput', 0);

for iCon = 1:size(dsCon{1},2)
    

    
    whichTrial = find(label(:,iCon)==1);
    
    for iTrial = 1:numel(whichTrial)
        beta{iCon,1}(iTrial,:) = nanmean(tmp(whichTrial(iTrial)+TR-3:whichTrial(iTrial)+TR-1,:));
        beta{iCon,2}(iTrial,:) = tmp(whichTrial(iTrial)+TR-1,:)-tmp(whichTrial(iTrial)+TR-3,:);
        beta{iCon,3}(iTrial,:) = tmp(whichTrial(iTrial)+TR-1,:);
    end
    

    
end


end