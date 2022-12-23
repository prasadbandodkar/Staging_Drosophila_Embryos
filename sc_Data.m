% This script prepares data in the 'data' folder for deep learning. It
% performs 2 tasks. First, it calculates the regression ids for each image
% in every folder and makes a csv file with the filename and regression
% ids. Second, it opens the metadata.txt file for each image and finds
% image dimension, dimension of the image in microns, and the scaling
% factor in microns/px. 

clear
clc
close all

foldername  = '/Volumes/Extreme/Projects/staging';
path_data   = [foldername, '/data'];
xls         = readtable('Xls/data.xlsx');
rawxls      = readtable('Xls/raw.xlsx');
xls         = table2struct(xls);
rawxls      = table2struct(rawxls);



% Scaling info in metadata: lsm, czi,lif
ftype = ["lsm","czi","lif"];
scalingstr = ["VoxelSizeX","Detector|ScaledImageRectangleSize", ...
                "DimensionDescription #1|Length"];



%
% Get regression ids and extract essential metadata into another csv
%
nXlsFiles     = length(xls);
microndim     = zeros(nXlsFiles,1);
scalingfactor = microndim;
imgdim        = microndim;
for i=1:nXlsFiles
    disp(['Running: ',num2str(i),'/',num2str(nXlsFiles)])

    filelocation = xls(i).filelocation;
    filetype     = xls(i).filetype;
    tstart       = xls(i).tstart;
    tend         = xls(i).tend;
    files        = dir(filelocation);
    files        = {files.name}';
    i0           = cellfun(@(x) contains(x,'png'),files,'UniformOutput',false);
    files        = files(cell2mat(i0));

    %
    % Task 1: Get regreesion ids
    %
    nFiles = length(files);
    id     = zeros(nFiles,1);
    for j = 1:nFiles
        c     = strsplit(files{j},{'_','.'});
        t     = str2double(c{end-1}(2:end));
        id(j) = (t - tstart)/(tend-tstart);
    end

    % Add ids to files and store as csv
    foldername  = strsplit(filelocation,'/');
    foldername  = foldername{end};
    files       = cellfun(@(x) [foldername,'/',x], files,'UniformOutput',false);
    files       = [files,num2cell(id)];
    writecell(files,[filelocation,'/id.csv'])



    %
    % Task 2: Metadata
    %
    metatxt  = strsplit(rawxls(i).filelocation,'.');
    metatxt  = [metatxt{1:end-1},'.',filetype,'.txt'];
    metacell = readcell(metatxt);

    if contains(filetype,'lsm')
        i0       = find(contains(string(metacell),'Height'));
        dim      = metacell{i0(1)};
        i0       = find(dim=='=');
        dim      = str2double(dim(i0+1:end));
        i0       = find(contains(string(metacell(:,1)),"VoxelSizeX"));
        scalingfactor(i) = metacell{i0(1),2};               % in microns/px
        microndim(i) = scalingfactor(i)*dim;                % in microns
        imgdim(i) = dim;                                    % in px    
        
    elseif contains(filetype,'czi')
        i0       = find(contains(string(metacell),'Height'));
        dim      = metacell{i0(1)};
        i0       = find(dim=='=');
        dim      = str2double(dim(i0+1:end));
        i0       = find(contains(string(metacell(:,1)),"Detector|ScaledImageRectangleSize"));
        miheight = metacell{i0(1),2};
        i0       = find(miheight==',');
        miheight = str2double(miheight(1:i0-1));
        microndim(i)  = miheight;                           % in microns
        scalingfactor(i) = miheight/dim;                    % in microns/px  
        imgdim(i)        = dim;                             % in px
    elseif contains(filetype,'lif')
        i0       = find(contains(string(metacell),'Height'));
        dim      = metacell{i0(1)};
        i1       = find(dim=='=');
        if isempty(i1)
            dim = metacell{i0(1),2};
        else
            dim = str2double(dim(i1+1:end));
        end
        i0       = find(contains(string(metacell(:,1)),"DimensionDescription #1|Length"));
        mheight  = metacell{i0(1)};
        i0       = find(mheight==':');
        miheight = str2double(mheight(i0+2:end))*10^6;      % in meters. m->micron
        microndim(i)     = miheight;                        % in microns
        scalingfactor(i) = miheight/dim;                    % in microns/px  
        imgdim(i)        = dim;                             % in px
    end
end

slno = [xls.slno]';
T = table(slno,imgdim,microndim,scalingfactor);
writetable(T,[path_data,'/metadata.csv',]);


































