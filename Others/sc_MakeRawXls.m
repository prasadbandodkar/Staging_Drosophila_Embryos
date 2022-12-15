import functions.*

foldername  = '/Volumes/Extreme/Projects/staging';

path_data = [foldername, '/raw/'];
datafiles = extractFileLocations(path_data,["lif","lsm","czi"]);
xlsfiles  = extractFileLocations(path_data,"xlsx");


%
% Get filenames and nc14 start and end points from the excel sheets
%
filenames = [];
nc14start = [];
nc14end   = [];
use       = [];
for i=1:length(xlsfiles)
    if ~contains(xlsfiles(i),'$')
        xls         = readtable(xlsfiles(i),'VariableNamingRule','preserve');
        filenames   = [filenames;string(xls.Filename)];
        nc14start   = [nc14start; xls.("nc 14 start")];
        nc14end     = [nc14end; xls.("nc 14 end")];
        if ismember('Use',xls.Properties.VariableNames)
            use = [use;xls.("Use")];
        else
            use = [use;ones(size(xls,1),1)];
        end
    end
end


%
% Get only those filenames where Use is true
%
use(isnan(use)) = 0;
use       = logical(use);
filenames = filenames(use);
nc14start = nc14start(use);
nc14end   = nc14end(use);


%
% Get filenames from xls and match with the corresponding data file
%
filelocation = [];
for i=1:length(filenames)
    file = filenames(i);
    file = strsplit(file,{'/','\'});
    file = file(end);
    i0 = contains(datafiles,file);
    if any(i0)
        filelocation = [filelocation; datafiles(i0)];
    else
        filelocation = [filelocation; ""];
    end
end


%
% Save as xls
%
T = table(filelocation, nc14start, nc14end, 'VariableNames',["filelocation", "nc14start", "nc14end"]);
writetable(T,'Data_raw.xlsx')




