clear
clc
close all

foldername  = '/Volumes/Extreme/Projects/staging';
path_data   = [foldername, '/data'];  


% get data files
folders = dir(path_data);


% read img sheet
imgxls = readtable('Xls/img.xlsx');
filelocation = [imgxls.filelocation];


%
% Loop over all folders
%
count = 1;
for i=1:length(folders)
    name = folders(i).name;
    path_datai = [folders(i).folder,'/',name];
    if ~(name(1) == '.')

        % make excel columns
        c  = strsplit(name,'_');
        c  = strjoin(c(3:end),'_');
        i0 = find(cell2mat(cellfun(@(x) contains(x,c),filelocation,'UniformOutput',false)));
        
        xls(count,:) = imgxls(i0,:);
        xls{count,"filelocation"} = {path_datai};
        count = count + 1;      
    end
end


% sort data according to slno
slno = [xls.slno];
[~,i0] = sort(slno);
xls = xls(i0,:);


% save data to excel sheet
writetable(xls,'Xls/data.xlsx')