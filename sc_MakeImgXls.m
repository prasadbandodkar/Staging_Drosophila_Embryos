clear
clc
close all

foldername  = '/Volumes/Extreme/Projects/staging';
path_img    = [foldername, '/img'];
path_data   = [foldername, '/data'];


%
% Get folder names and filenames
%
folders = dir(path_img);



%
% Loop over all folders
%
for i=1:length(folders)
    name = folders(i).name;
    path_imgi = [folders(i).folder,'/',name];
    if ~(name(1) == '.')
        
        % make destination folder
        c = strsplit(name,'_');
        path_datai = [path_data,'/',name];
        mkdir(path_datai)

        % make excel columns

        
    end
end



function [s,c,z,t] = getinfo(name)





end



