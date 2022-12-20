function Img2Data(path_img,path_data,sstart,send,tstart,tend,ch)

    files = dir(path_img);

    for i=1:length(files)
       name = files(i).name;
       if contains(name,'png')
            split = strsplit(name,{'_','.'});
            s = str2double(split{1}(2:end));
            c = str2double(split{2}(2:end));
            t = str2double(split{4}(2:end));
            if s>=sstart && s<=send && t>=tstart && t<=tend && c==ch
                path_datai = [path_data,'/',name];
                path_imgi  = [files(i).folder,'/',files(i).name];
                copyfile(path_imgi,path_datai)
            end
       end
       if contains(name,'csv')
            path_datai = [path_data,'/',name];
            path_imgi  = [files(i).folder,'/',files(i).name]; 
            copyfile(path_imgi,path_datai)
       end
    end
end

