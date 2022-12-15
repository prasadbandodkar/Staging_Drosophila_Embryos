clear
clc
close all


foldername  = '/Volumes/Extreme/Projects/staging';
path_img    = [foldername, '/img'];
path_data   = [foldername, '/data'];



folders     = dir(path_img);


%
% Loop over all files
%
nFiles       = length(folders);
for i=1:nFiles
   fprintf('Running: %i/%i \t\n',i,nFiles)

   % path of input img file
   imgfilename = folders(i).name;
   path_imgi = [folders(i).folder,'/',imgfilename];
   
   % path of output data files
   c   = strsplit(path_imgi,{filesep,'.'});
   i0  = find(strcmp(c,'img'));
   path_datai = [path_data,'/',num2str(i),'_',c{i0+1},'_',c{end-1}];
   mkdir(path_imgi) 


   sstart    = xls(i).sstart;
   tstart    = xls(i).tstart; 
   send      = xls(i).send;
   tend      = xls(i).tend;
   ch        = xls(i).ch;
   makeTrainValTestData(filelocation,sstart,tstart,send,tend,ch);
end





 