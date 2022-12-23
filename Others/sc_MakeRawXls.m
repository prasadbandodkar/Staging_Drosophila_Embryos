
foldername  = '/Volumes/Extreme/Projects/staging';

path_org = [foldername, '/org/'];
path_raw = [foldername, '/raw/'];
datafiles = extractFileLocations(path_org,["lif","lsm","czi"]);
xlsfiles  = extractFileLocations(path_org,"xlsx");


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
nFiles = length(filenames);
for i=1:nFiles
    disp(['running: ',num2str(i),'/',num2str(nFiles)])
    file = filenames(i);
    file = strsplit(file,{'/','\'});
    file = file(end);
    i0 = contains(datafiles,file);
    if any(i0)
        c = char(datafiles(i0));
        c = strsplit(c,'/');
        c = c{end};
        c(c==' ' | c==';') = '_';
        file  = string([path_raw,num2str(i),'_',char(c)]);
         copyfile(datafiles(i0),file)
        filelocation = [filelocation; file];
    else
        filelocation = [filelocation; ""];
    end
end


% move slno to first column
xls = movevars(xls,'slno','Before','filelocation');


%
% Save as xls
%
T = table(filelocation, nc14start, nc14end, 'VariableNames',["filelocation", "nc14start", "nc14end"]);
writetable(T,'raw.xlsx')




