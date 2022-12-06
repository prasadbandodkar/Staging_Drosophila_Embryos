clear
clc
close all


foldername  = '/Volumes/Extreme/Projects/staging';
path_data   = [foldername, '/raw/'];
nInterp     = 10;
scaleTo     = 4.4e-7;
xls         = readtable('Data.xlsx');
xls         = table2struct(xls);



%
% Loop over all files
%
nFiles = length(xls);
for i=1:nFiles
   fprintf('Running: %i/%i \t\n',i,nFiles)
   filelocation = xls(i).filelocation;
   nc14start    = xls(i).nc14start;
   nc14end      = xls(i).nc14end;
   makeTrainValTestData(filelocation,nc14start,nc14end,path_data,scaleTo,nInterp);
end





 