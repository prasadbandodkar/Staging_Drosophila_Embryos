clear
clc


foldername  = '/Volumes/Extreme/Projects/staging';

path_data = [foldername, '/raw'];
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
   path_raw = xls(i).filelocation;
   
   
   % Get an image location based on filelocation
   c   = strsplit(path_raw,{filesep,'.'});
   i0  = find(strcmp(c,'raw'));
   path_imgi = [path_img,'/',num2str(i),'_',c{i0+1},'_',c{end-1}];
   mkdir(path_imgi) 
   

   % Get raw images
   Raw2Tiff(path_raw,path_imgi);
end
