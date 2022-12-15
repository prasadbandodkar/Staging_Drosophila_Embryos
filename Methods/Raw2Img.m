function Raw2Img(path_raw,path_img,printimg,printmeta)

arguments
    path_raw  = [];
    path_img  = [];
    printimg  = true;
    printmeta = true;
end

warning('off', 'MATLAB:MKDIR:DirectoryExists');

%
% Read image from bioformats
%
[IM,imdata,metadata] = readBioImage(path_raw);


nSeries = length(imdata);
if printimg
    %
    % Get path names for images: s/c#z#t#
    %
    path_imgp = cell(nSeries,1);
    for i=1:nSeries
        chnum           = imdata{i}.chnum;
        znum            = imdata{i}.znum;
        tnum            = imdata{i}.tnum;
        nImages         = length(chnum);
        path_imgp{i}    = cell(nImages,1);
        for j=1:nImages
            path_imgp{i}{j} = [path_img,'/s',num2str(i),'_c',num2str(chnum(j)),'_z',num2str(znum(j)),'_t',num2str(tnum(j)),'.png'];
        end
    end
    %
    % Print images to folder
    %
    for i=1:nSeries
        IMs         = IM{i};
        path_imgs   = path_imgp{i};
        nImages     = length(IMs);
        for j = 1:nImages
            I =  mat2gray(IMs{j});
            imwrite(I, [path_imgs{j}])
        end
    end
end


%
% Save metadata to csv file
%
if printmeta
    metacsv    = [];
    count      = 1;
    for j=1:nSeries
        data = char(metadata{j});
        data(data=='{' | data=='{') = [];
        data        = strsplit(data,',');
        for i = 1:length(data)
            c = data{i};
            c(c=='|') = '.';
            c(c=='_' | c=='#' | c=='-') = [];
            if c(1) == ' '
                c(1) = [];
            end
            c(c==' ') = '_';
            c = strsplit(c,'=');
            if length(c) == 2
                metacsv{count,1} = c{1};
                metacsv{count,2} = c{2};
                count            = count + 1;
            end   
        end
        writecell(metacsv,[path_img,'/s',num2str(j),'_metadata.csv'])
    end  
end
    
end


















