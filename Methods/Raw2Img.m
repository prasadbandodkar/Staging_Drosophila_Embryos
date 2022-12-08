function Raw2Tiff(path_raw,path_img)

warning('off', 'MATLAB:MKDIR:DirectoryExists');

[IM,imdata,metadata] = readBioImage(path_raw);


nSeries = length(imdata);



%
% Get path names for images: s/c#z#t#
%
path_imgp = cell(nSeries,1);
for i=1:nSeries

    path_series = [path_img,'/s',num2str(i)];
    mkdir(path_series)

    chnum = imdata{i}.chnum;
    znum  = imdata{i}.znum;
    tnum  = imdata{i}.tnum;
    nImages = length(chnum);

    path_imgp{i} = cell(nImages,1);
    for j=1:nImages
        path_imgp{i}{j} = [path_series,'/c',num2str(chnum(j)),'_z',num2str(znum(j)),'_t',num2str(tnum(j)),'.png'];
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


%
% Save metadata to csv file
%
