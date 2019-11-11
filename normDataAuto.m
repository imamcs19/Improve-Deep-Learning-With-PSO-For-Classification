clc
clear all
close all

%% Parameter-parameter yang digunakan 
% 
% untuk normalisasi nilai dari fitur
max = 300;
min = 0;
max2 = 1;
min2 = 0;
%bykData = 8;
k = 3; % banyaknya padding, ukuran matrik filter (k x k)
byk_fitur=4;
windows_size=2; % sebagai filter [windows_size x windows_size]
ws=windows_size;

% set ukuran matrik square untuk dataset
size_sq=5;
%%


%% load data awal (1D)
[dataAwal,txt,raw] = xlsread('data.xlsx');
bykData = numel(dataAwal)-byk_fitur;
%%

%% membentuk pola-pola data dengan 4 fitur (4D) + 1 target (1D)

% gandakan data awal sebanyak byk_fitur + 1
repmatData=repmat(dataAwal,1,byk_fitur+1);

% get data yang sudah berupa 4 fitur +  1 target
% x1, x2, x3, x4, y
for i=1:bykData
    %diag(repmatData(i:end,:))'
    % data fitur plus target
    dataset{i}=diag(repmatData(i:end,:))';
    
    % hanya data fitur saja
    data{i}=dataset{i}(1:byk_fitur);
end


%% opsi 1: membentuk dataset sebagai matriks square 5x5 menggunakan konsep 
% imresize(dataset{i},[size_sq size_sq])
for i=1:bykData
    a{i}=imresize(data{i},[size_sq size_sq]);
end
%%

%% opsi 2: membentuk dataset sebagai matriks square 5x5 menggunakan konsep spiral
% a{1} = [
% 0	0	0	0	0
% 0	0	131	259	0
% 0	0	253	95	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
% a{2} = [0	0	0	0	0
% 0	0	259	95	0
% 0	0	131	263	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
% a{3} = [0	0	0	0	0
% 0	0	95	263	0
% 0	0	259	50	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
% a{4} = [0	0	0	0	0
% 0	0	263	50	0
% 0	0	95	154	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
% a{5} = [0	0	0	0	0
% 0	0	50	154	0
% 0	0	263	73	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
% a{6} = [0	0	0	0	0
% 0	0	154	73	0
% 0	0	50	141	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
% a{7} = [0	0	0	0	0
% 0	0	73	141	0
% 0	0	154	146	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
% a{8} = [0	0	0	0	0
% 0	0	141	146	0
% 0	0	73	87	0
% 0	0	0	0	0
% 0	0	0	0	0
% ];
%%

% a1 = a{1};
% a2 = a{2};
% a3 = a{3};
% a4 = a{4};
% a5 = a{5};
% a6 = a{6};
% a7 = a{7};
% a8 = a{8};

% norm{1} = (((a1-min)./(max-min))*(max2-min2))+min2;
% norm{2} = (((a2-min)./(max-min))*(max2-min2))+min2;
% norm{3} = (((a3-min)./(max-min))*(max2-min2))+min2;
% norm{4} = (((a4-min)./(max-min))*(max2-min2))+min2;
% norm{5} = (((a5-min)./(max-min))*(max2-min2))+min2;
% norm{6} = (((a6-min)./(max-min))*(max2-min2))+min2;
% norm{7} = (((a7-min)./(max-min))*(max2-min2))+min2;
% norm{8} = (((a8-min)./(max-min))*(max2-min2))+min2;

%% menghitung hasil norm
for i=1:bykData
    norm{i}=(((a{i}-min)./(max-min))*(max2-min2))+min2;
end
%%

% simpan hasil norm
save('afile.mat','norm')

% b1 = norm{1}
% b2 = norm{2};
% b3 = norm{3};
% b4 = norm{4};
% b5 = norm{5};
% b6 = norm{6};
% b7 = norm{7};
% b8 = norm{8};

% imwrite(b1,'1.jpg')
% imwrite(b2,'2.jpg')
% imwrite(b3,'3.jpg')
% imwrite(b4,'4.jpg')
% imwrite(b5,'5.jpg')
% imwrite(b6,'6.jpg')
% imwrite(b7,'7.jpg')
% imwrite(b8,'8.jpg')

%% simpan hasil norm dalam bentuk citra
for i=1:bykData
    imwrite(norm{i},strcat(num2str(i),'norm.jpg'));
end
%%

% misal ada 3 filter convolution yang digunakan:
% ke-1 (conv11) : average filter
% ke-2 (conv12) : max filter
% ke-3 (conv13) : std filter

% % melakukan convolution dengan average filter, max filter,
% % std filter
% d1_conv11=Function_AvgFilter(b1,3)
% d1_conv12=Function_MaxFilter_(b1,3)
% d1_conv13=Function_STDFilter(b1,3)

%% melakukan convolution dengan average filter, max filter,
% std filter secara iteratif
for i=1:bykData
    d1_conv11AvgFilter{i}=Function_AvgFilter(norm{i},k);
    d1_conv12MaxFilter{i}=Function_MaxFilter_(norm{i},k);
    d1_conv13STDFilter{i}=Function_STDFilter(norm{i},k);
end
%%

%% simpan hasil convolution dalam bentuk citra
for i=1:bykData
    imwrite(d1_conv11AvgFilter{i},strcat(num2str(i),'convAvg.jpg'));
    imwrite(d1_conv12MaxFilter{i},strcat(num2str(i),'convMax.jpg'));
    imwrite(d1_conv13STDFilter{i},strcat(num2str(i),'convSTD.jpg'));
end
%%

%% melkukan pooling pada setiap hasil conv. dari average filter, 
% max filter,std filter secara iteratif
for i=1:bykData
    d1_pool11AvgFilter{i}=Function_Pooling(d1_conv11AvgFilter{i},ws);
    d1_pool12MaxFilter{i}=Function_Pooling(d1_conv12MaxFilter{i},ws);
    d1_pool13STDFilter{i}=Function_Pooling(d1_conv13STDFilter{i},ws);
end
%%

%% simpan hasil pooling dalam bentuk citra
for i=1:bykData
    imwrite(d1_pool11AvgFilter{i},strcat(num2str(i),'poolAvg.jpg'));
    imwrite(d1_pool12MaxFilter{i},strcat(num2str(i),'poolMax.jpg'));
    imwrite(d1_pool13STDFilter{i},strcat(num2str(i),'poolSTD.jpg'));
end
%%

disp("Done......!");














