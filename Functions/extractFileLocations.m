function FileLocations = extractFileLocations(directory,ext,checkSubFolder)
% This functions extracts all ".ext" files from a given parent directory 
%
% Input :
%       "directory": character/string vector contains name of parent 
%                     directory to search for ".ext" files
%       "ext": Searches for files with the extension ".ext"
%       "checksubFolder" (optional): accepts boolean values- true or false
%                       If yes, the function includes subfolders in the
%                       directory. Default is true.
%
% Output :
%       "FileLocations": String column vector that contains file locations

if ~exist('ext','var')
   ext = "lsm";
end
if ~exist('checkSubFolder','var')
   checkSubFolder = true;
end
if ischar(ext)
    ext = string(ext);
end


% check subfolders if asked for
if checkSubFolder
    midstr = '/**/*.';
else
    midstr = '/*.';
end


% get filenames
a = [];
for i=1:length(ext)
    ext1 = ext(i);
    c    = strsplit(ext1,".");
    ext1 = char(c(end));
    a    = [a; dir([directory,midstr,ext1])];
end


a             = struct2table(a);
FileLocations = string(strcat(a.folder,'/', a.name));

end



