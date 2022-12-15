function Img2Data(filelocation,nc14start,nc14end,path_data,scaleTo,nInterp)

%
% Read image
%
[IM,metadata] = imread(filelocation)


%
% Image pre-processing
%
IM            = cleanImage(IM,numch);            % Subtract background
IM            = squeeze(IM(:,:,nucch,:));
IM            = reshape(IM,H,W,nZstacks,[]);
c             = strsplit(fileLocation,{'/','\','.'});
filename      = [num2str(slno),'_',char(c(end-1))]; 


%
% Get metadata
%


%
% Loop
%
for i=nc14start:nc14end
    istr  = num2str(i);
    numID = (i-ncstart)/(ncend - ncstart);



    
end




end

