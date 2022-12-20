% This script will read the data from the "img" folder and the "raw.xlsx"
% sheet and generate a "img.xlsx" that stores the same data as the raw
% sheet but with the file locations of the images in the img folder. 

clear
clc
close all

foldername  = '/Volumes/Extreme/Projects/staging';
path_img    = [foldername, '/img'];


% get img files
folders = dir(path_img);


% read raw xls sheet 
rawxls = readtable('Xls/raw.xlsx');
filelocation = [rawxls.filelocation];


%
% Loop over all folders
%
count = 1;
for i=1:length(folders)
    name = folders(i).name;
    path_imgi = [folders(i).folder,'/',name];
    if ~(name(1) == '.')

        % make excel columns
        c  = strsplit(name,'_');
        c  = strjoin(c(3:end),'_');
        i0 = find(cell2mat(cellfun(@(x) contains(x,c),filelocation,'UniformOutput',false)));
        
        xls(count,:) = rawxls(i0,:);
        xls{count,"filelocation"} = {path_imgi};
        count = count + 1;      
    end
end


% sort data according to slno
slno = [xls.slno];
[~,i0] = sort(slno);
xls = xls(i0,:);


% save data to excel sheet
writetable(xls,'Xls/img.xlsx')








