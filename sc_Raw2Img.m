clear
clc


foldername  = '/Volumes/Extreme/Projects/staging';

path_raw  = [foldername, '/raw'];
path_img  = [foldername, '/img'];  
xls       = readtable('raw.xlsx');
xls       = table2struct(xls);


warning('off', 'MATLAB:MKDIR:DirectoryExists');


%
% Loop over all files
%
nFiles = length(xls);
tic
for i=1:nFiles
   fprintf('Running: %i/%i \t\n',i,nFiles)

   % path of input raw file
   path_rawi = xls(i).filelocation;
   
   
   % path of output image files
   c   = strsplit(path_rawi,{filesep,'.'});
   i0  = find(strcmp(c,'raw'));
   path_imgi = [path_img,'/',c{end-1},'_',c{end}];
   mkdir(path_imgi) 
  

   % Get raw images
   Raw2Img(path_rawi,path_imgi,true,true);
end
toc
