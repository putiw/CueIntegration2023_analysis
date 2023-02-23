function beta = getTR(datafiles,dsCon,whichRoi,TR)

% given samples and design matrix

beta = cell(size(dsCon{1},2),1);

for iCon = 1:size(dsCon{1},2)
    
    
    
    tmp = cellfun(@(m)m(whichRoi,:), datafiles, 'UniformOutput', 0);
    tmp = cellfun(@(m)100*(m-nanmean(m,2))./nanmean(m,2), tmp, 'UniformOutput', 0);
    tmp = cat(2,tmp{:})';
    
    label = cat(1,dsCon{:});
    
    beta{iCon} = zeros(sum(label(:,iCon)),size(tmp,2));
    
    whichTrial = find(label(:,iCon)==1);
    
    for iTrial = 1:numel(whichTrial)
        beta{iCon}(iTrial,:) = tmp(whichTrial(iTrial)+TR-1,:);
    end
    
    
end

end