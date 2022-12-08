clear
clc


foldername  = '/Volumes/Extreme/Projects/staging';

path_raw  = [foldername, '/raw'];
path_img  = [foldername, '/img'];  
xls       = readtable('Data.xlsx');
xls       = table2struct(xls);


warning('off', 'MATLAB:MKDIR:DirectoryExists');


%
% Loop over all files
%
nFiles = length(xls);
for i=1:nFiles
   fprintf('Running: %i/%i \t\n',i,nFiles)

   % path of input raw file
   path_rawi = xls(i).filelocation;
   
   
   % path of output image files
   c   = strsplit(path_raw,{filesep,'.'});
   i0  = find(strcmp(c,'raw'));
   path_imgi = [path_img,'/',num2str(i),'_',c{i0+1},'_',c{end-1}];
   mkdir(path_imgi) 
   

   % Get raw images
   Raw2Img(path_rawi,path_imgi);
end
