clear
clc
close all


foldername  = '/Volumes/Extreme/Projects/staging';
path_img    = [foldername, '/img'];
path_data   = [foldername, '/data'];
imgext      = 'png';


warning('off', 'MATLAB:MKDIR:DirectoryExists');


%
% Load img xls files
%
xls = readtable('Xls/img.xlsx');



%
% Loop over all files
%
nFiles = size(xls,1);
for i = 1:nFiles

   fprintf('Running: %i/%i \t\n',i,nFiles)

   %
   % get all the image filenames within the folder and create data folder
   %
   path_imgi    = cell2mat(xls{i,"filelocation"});
   foldername   = strsplit(path_imgi,'/');
   foldername   = foldername{end};
   path_datai   = [path_data,'/',foldername];


   %
   % copy the files that meet s,c,z,t criteria
   %
   sstart = xls{i,"sstart"};
   send   = xls{i,"send"};
   tstart = xls{i,"tstart"};
   tend   = xls{i,"tend"};
   ch     = xls{i,"ch"};

   %
   % Make folder and copy files
   %
   mkdir(path_datai)
   Img2Data(path_imgi,path_datai,sstart,send,tstart,tend,ch)
   
end





 