clc
clear all
close all
warning off

bykKelas = 4
byk_kulit_kering = 30
byk_kulit_sehat = 30
byk_mata_katarak = 23
byk_mata_normal = 18

bykData = 101;
ukuranResize =32;
bykFitur = ukuranResize*ukuranResize;

data=zeros(bykData,bykFitur+1);
%load data kelas kulit kering (30 data)
for t=1:byk_kulit_kering
    nama_file = strcat('crops_kulit_kering/',num2str(t),'.jpg');
    %disp(nama_file)
    dataTemp=rgb2gray(imresize(imread(nama_file),[ukuranResize ukuranResize]));
    data(t,1:bykFitur)=dataTemp(:)';
    data(t,bykFitur+1)= 1;
end

%load data kelas kulit sehat (30 data)
for t=1:byk_kulit_sehat
    nama_file = strcat('crops_kulit_sehat/',num2str(t),'.jpg');
    %disp(nama_file)
    dataTemp=rgb2gray(imresize(imread(nama_file),[ukuranResize ukuranResize]));
    data(t+byk_kulit_kering,1:bykFitur)=dataTemp(:)';
    data(t+byk_kulit_kering,bykFitur+1)= 2;
end

%load data kelas mata katarak (23 data)
for t=1:byk_mata_katarak
    nama_file = strcat('crops_mata_katarak/',num2str(t),'.jpg');
    %disp(nama_file)
    dataTemp=rgb2gray(imresize(imread(nama_file),[ukuranResize ukuranResize]));
    data(t+byk_kulit_kering+byk_kulit_sehat,1:bykFitur)=dataTemp(:)';
    data(t+byk_kulit_kering+byk_kulit_sehat,bykFitur+1)= 3;
end

%load data kelas mata normal (18 data)
for t=1:byk_mata_normal
    nama_file = strcat('crops_mata_normal/',num2str(t),'.jpg');
    %disp(nama_file)
    dataTemp=rgb2gray(imresize(imread(nama_file),[ukuranResize ukuranResize]));
    data(t+byk_kulit_kering+byk_kulit_sehat+byk_mata_katarak,1:bykFitur)=dataTemp(:)';
    data(t+byk_kulit_kering+byk_kulit_sehat+byk_mata_katarak,bykFitur+1)= 4;
end

%save data to excel
xlswrite('datatrainCitraClassify.xlsx',data);
xlswrite('datatestCitraClassify.xlsx',data);

disp('Done.....!')