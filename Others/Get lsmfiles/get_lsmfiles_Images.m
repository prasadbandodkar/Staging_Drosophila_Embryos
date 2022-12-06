%% Get locations of all files in the folder 
clear
clc
close all

lsmfolder = 'G:\.shortcut-targets-by-id\1_zpEK73II9qaZ0GnBQCTze5eizzauuHS\Confocal';
datfolder = './Data/';
Filenames = extractFileLocations(lsmfolder,'lsm',true);


%% Loop over the list to find files of interest

v = false(length(Filenames),1);
for i=1:length(Filenames)
   filename = char(Filenames(i)); 
   vmatch = any(regexp(filename,'H2A')) & any(regexp(filename,'timecourse'));       % Make changes here!
   if vmatch
      v(i) = true; 
   end
end

foundfiles = Filenames(v);

%% Copy lsm files to its own folder

for i=1:length(foundfiles)
   filename = char(foundfiles(i));
   name = strsplit(filename,{filesep,'.'});
   name = [char(name(4)),'_',char(name(5)),'_',char(name(6)),'.lsm'];
   destination = [datfolder,name];
   copyfile(filename,destination,'f')  
end