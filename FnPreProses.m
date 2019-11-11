function [bykData,byk_fitur,target,norm]=FnPreProses(...
    namafile, max, min, max2, min2)

%% load data (5D), sebagai data training atau testing
[dataAwal,txt,raw] = xlsread(namafile);
bykData = size(dataAwal,1);
byk_fitur = size(dataAwal,2)-1;
%%

% get data yang sudah berupa 4 fitur (4D) +  1 target (1D)
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
ukuranSize=sqrt(numel(data{1}));
for i=1:bykData
    %a{i}=repmat(data{i},[byk_fitur 1]);
    %data{i}
    
    %pause(5000)
	a{i}=reshape(data{i},[ukuranSize ukuranSize]);
end
% misal vektor (253   131   259    95), menjadi matrik square
%  (                         )
%  (  253   131   259    95  )
%  (  253   131   259    95  )
%  (  253   131   259    95  )
%  (  253   131   259    95  )
%  (                         )
%%

% %% opsi 2: membentuk dataset sebagai matriks square 
% [byk_fitur x byk_fitur] menggunakan konsep 
% % imresize(dataset{i},[byk_fitur byk_fitur])
% for i=1:bykData
%     a{i}=imresize(data{i},[size_sq size_sq]);
% end
% %%

%% opsi 3: membentuk dataset sebagai matriks square
% [byk_fitur x byk_fitur] menggunakan konsep spiral
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

% a{1}
% norm{1}
% 
% pause(50000000)
%%

% % simpan hasil norm
% save('afile.mat','norm')
% 
% %% simpan hasil norm dalam bentuk citra
% for i=1:bykData
%     imwrite(norm{i},strcat(num2str(i),'norm.jpg'));
% end
% %%

% misal ada 3 filter convolution yang digunakan:
% ke-1 (conv11) : average filter
% ke-2 (conv12) : max filter
% ke-3 (conv13) : std filter
    
    
    