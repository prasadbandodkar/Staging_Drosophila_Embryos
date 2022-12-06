clear
clc
close all

folder = './lsmfiles/Alllsmfiles/dorsal dosage good files/';
datfolder = './lsmfiles/Dorsal_new/';

load([folder,'filenames_dorsal.mat'])

for i=1:length(filenames_dorsal)
    
   filename = char(filenames_dorsal(i));
   filename = [folder,filename];
   name = strsplit(filename,{filesep,'.'});
   name = [char(name(4)),'_',char(name(5)),'_',char(name(6)),'.lsm'];
   destination = [datfolder,name];
   copyfile(filename,destination,'f')  
    
end