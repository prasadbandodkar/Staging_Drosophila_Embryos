function [IM,imdata,metadata] = readBioImage(filelocation)


%
% Open image file using bfmatlab
%
imraw = bfopen(filelocation);
imraw = imraw(:,1:2);



%
% Get Images, info and metadata for every series
%
nSeries = size(imraw,1);
IM = cell(nSeries,1);   info = IM;  metadata = IM;  imdata = IM;
for i=1:nSeries
    IM{i}          = imraw{i}(:,1);
    info{i}        = imraw{i}(:,2);
    nImg = length(info{i});
    chnum = zeros(nImg,1);  znum = chnum;   tnum = chnum;
    for j=1:length(info{i})
        in       = info{i}{j};
        chnum(j) = findvalue(in,'C');
        znum(j)  = findvalue(in,'Z');
        tnum(j)  = findvalue(in,'T');
    end
    imdata{i}.chnum = chnum;
    imdata{i}.znum  = znum;
    imdata{i}.tnum = tnum;
    metadata{i}    = char(imraw{i,2});
end




%
% Get Metadata
%
% metadata    = [];
% count       = 1;
% for j=1:nSeries
%     data = char(meta{j});
%     data(data=='{' | data=='{') = [];
%     data        = strsplit(data,',');
%     for i=1:length(data)
%         c = data{i};
%         c(c=='|') = '.';
%         c(c=='_' | c=='#' | c=='-') = [];
%         if c(1)==' '
%             c(1) = [];
%         end
%         c(c==' ') = '_';
%         c = strsplit(c,'=');
%         if length(c) == 2
%             metadata{count,1} = c{1};
%             metadata{count,2} = c{2};
%             count = count + 1;
%         end   
%     end
% end

end


function n = findvalue(in,c)

i0 = find(in=='.');
in = in(i0+1:end);
in(in==';') = [];
in(in==' ') = '_';

if contains(in,c) 
    i0 = find(in==c);
    i0 = i0(end);
    il = i0 + find(in(i0:end)=='=');
    il = il(1);
    ir = i0 + find(in(i0:end)=='/') - 2;
    ir = ir(1);
    if isempty(ir)
        ir = length(in);
    end
    n = str2double(in(il:ir));
else
    n = 1;
end

end
















