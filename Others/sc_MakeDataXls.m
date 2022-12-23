clear
clc
close all

foldername  = '/Volumes/Extreme/Projects/staging';
path_data   = [foldername, '/data'];  

warning('off')

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
        filetype = c{end};
        c  = strjoin(c(1:end-1),'_');
        i0 = find(cell2mat(cellfun(@(x) contains(x,c),filelocation,'UniformOutput',false)));
        
        if count==1
            xls(count,:) = imgxls(i0,:);
        else
            xls(count,1:end-1) = imgxls(i0,:);
        end
        xls{count,"filelocation"} = {path_datai};
        xls{count,"filetype"} = filetype;
        
        count = count + 1;      
    end
end


% sort data according to slno
slno = [xls.slno];
[~,i0] = sort(slno);
xls = xls(i0,:);

% move slno to first column
xls = movevars(xls,'slno','Before','filelocation');


% save data to excel sheet
writetable(xls,'Xls/data.xlsx')