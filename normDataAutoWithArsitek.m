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

%byk_fitur=4;
%bykData = 8;

% misal ada 3 filter convolution yang digunakan:
% ke-1 (conv11) : average filter
% ke-2 (conv12) : max filter
% ke-3 (conv13) : std filter
bykFilter = 3;


% banyaknya padding, ukuran matrik filter (k x k) pada proses
% convolution
k = 3; 

% sebagai filter [windows_size x windows_size]
% pada proses pooling
windows_size=2;
ws=windows_size;

%% set ukuran matrik square untuk dataset
% size_sq=5;
% %%
% atau dengan men-set size_sq=byk_fitur;


%% load data awal (5D)
[dataAwal,txt,raw] = xlsread('dataClassify.xlsx');
bykData = size(dataAwal,1);
byk_fitur = size(dataAwal,2)-1;
%%

%% membentuk pola-pola data dengan 4 fitur (4D) + 1 target (1D)

% gandakan data awal sebanyak byk_fitur + 1
% repmatData=repmat(dataAwal,1,byk_fitur+1);

% get data yang sudah berupa 4 fitur +  1 target
% x1, x2, x3, x4, y
for i=1:bykData
    %diag(repmatData(i:end,:))'
    % data fitur plus target
    dataset{i}=dataAwal(i,:);
    
    % hanya data fitur saja
    data{i}=dataset{i}(1:byk_fitur);
    
    % target atau nilai prediksi
    % 1 artinya kelas tinggi, 2 artinya kelas sedang, 3 artinya kelas
    % rendah
    target(i)=dataset{i}(end);
end

%% opsi 1: membentuk dataset sebagai 
% matriks square [byk_fitur x byk_fitur]
% menggunakan konsep replika matriks repmat(data{i},[byk_fitur 1])
for i=1:bykData
    a{i}=repmat(data{i},[byk_fitur 1]);
end
%%

% %% opsi 2: membentuk dataset sebagai matriks square 5x5 menggunakan konsep 
% % imresize(dataset{i},[size_sq size_sq])
% for i=1:bykData
%     a{i}=imresize(data{i},[size_sq size_sq]);
% end
% %%

%% opsi 3: membentuk dataset sebagai matriks square 5x5 menggunakan konsep spiral
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

%% menghitung hasil norm
for i=1:bykData
    norm{i}=(((a{i}-min)./(max-min))*(max2-min2))+min2;
end
%%

% simpan hasil norm
save('afile.mat','norm')

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

%% =============== menggunakan CNN no.1-no.8
%% =============== menggunakan Deep Learning no.9-no.11
%%                 karena hiddennya > 3 layer
%% =============== misal menggunakan arsitek:
%% 1. "Convolution"
%% 2. "Sigmoid/ReLU/lainnya"
%% 3. "Convolution"
%% 4. "Sigmoid/ReLU/lainnya"
%% 5. "Pooling"
%% 6. "Convolution" 
%% 7. "Sigmoid/ReLU/lainnya"
%% 8. "Pooling"
%% "9. Fully connected"
%% "10. Fully connected"
%% "11. Fully connected"

% dimana "Fully connected" dapat menggunakan Backpro atau ELM
% jadi sebelum masuk "Fully connected" maka harus disiapkan 
% dalam bentuk vektor (misal 1 baris banyak kolom) yg merupakan
% gabungan dari beberapa pooling
% misal disini menggunakan ELM

% ukuran setiap hasil [bykFilter x bykData]
hC=FnConvDL(norm,bykData,k) % 1. Convolution
hA=FnSigDL(hC,bykFilter,bykData)   % 2. Aktivasi
hC=FnConvInDL(hA,bykData,k,bykFilter) % 3. Convolution In
hA=FnSigDL(hC,bykFilter,bykData)   % 4. Aktivasi
hP=FnPoolDL(hA,windows_size,bykFilter,bykData)   % 5. Pooling
hC=FnConvInDL(hP,bykData,k,bykFilter) % 6. Convolution In
hA=FnSigDL(hC,bykFilter,bykData)   % 7. Aktivasi
windows_size=1;
hP=FnPoolDL(hA,windows_size,bykFilter,bykData)   % 8. Pooling

% 9. Fully connected ke-1, misal dengan ELM
byk_neuron_hidden_layer=5;
[hFC1,W1,Bias1,Beta1]=FnELMClassify(hP,target,...
    byk_neuron_hidden_layer,bykData,bykFilter);

% 10. Fully connected ke-2, misal dengan ELM
byk_neuron_hidden_layer=7;
[hFC2,W2,Bias2,Beta2]=FnELMInClassify(hFC1,target,...
    byk_neuron_hidden_layer,bykData,bykFilter);

% 11. Fully connected ke-3, misal dengan ELM
byk_neuron_hidden_layer=4;
[hFC3,W3,Bias3,Beta3]=FnELMInClassify(hFC2,target,...
    byk_neuron_hidden_layer,bykData,bykFilter);
% [hFC1,hFC2]=FnELMClassify(Xtrain,Ytrain,byk_neuron_hidden_layer,...
%     Xtest,Ytest,bykData,bykFilter)


%% Proses Testing 
% ketika proses testing, maka melalui proses CNN, lalu baru ""

disp("Done......!");














