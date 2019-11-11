function [Result]=FnPoolDL(hA,windows_size,bykFilter,bykData)

for ii=1:bykFilter
    for jj=1:bykData
        % ukuran baris dan kolom dari I
        I=hA{ii}{jj};
        [mI,nI]=size(I);

        % hitung pad yang dibutuhkan
        % padX=abs(2*ceil(nI/windows_size)-nI)
        padX=(ceil(nI/windows_size)*windows_size)-nI;
        % padY=abs(2*ceil(mI/windows_size)-mI)
        padY=(ceil(mI/windows_size)*windows_size)-mI;

        % inisialisasi matriks Ipad
        Ipad = zeros((mI+padY),(nI+padX));

        % inisialisasi vektor X={x1,x2,.....,xN} untuk hitung hasil
        % tiap elemen matrik pooling dari nilai max tiap barisnya
        % X_=zeros((mI+padY)*(nI+padX)/windows_size^2,windows_size^2);
        % size(X_);

        % besarnya ukuran matrik hasil pooling
        mpoolI=sqrt((mI+padY)*(nI+padX)/windows_size^2);
        npoolI=sqrt((mI+padY)*(nI+padX)/windows_size^2);
        
        % pause(5000000)

        % inisialisasi matrik penampung I
        Ipad(1:mI,1:nI)=I;
        
        % pause(5000000)

        % mengambil tiap bagian Ipad dgn size [windows_size x windows_size]
        % untuk membentuk matrik Ipool
        ws=windows_size;
        for i=1:mpoolI
            for j=1:npoolI
                %Ipool(i,j)=max(Ipad(i:i+(ws-1),j:j+(ws-1)));
                %1 (1:2,1:2)  |  2 (1:2,3:4)
                %i*npoolI+j
                %[((i-1)*ws + 1)  (((i-1)*ws + 1)+(ws-1))  ((j-1)*ws + 1)  (((j-1)*ws + 1)+(ws-1)) ]
                %(i-1)*npoolI+(j-1)+1 = {1,2,....,mpoolI*npoolI}
                Mpool{(i-1)*npoolI+(j-1)+1}=Ipad(((i-1)*ws + 1):(((i-1)*ws + 1)+(ws-1)),((j-1)*ws + 1):(((j-1)*ws + 1)+(ws-1)));
                Ipool(i,j)= max(max(Mpool{(i-1)*npoolI+(j-1)+1}));
            end
        end
        
        Result{ii}{jj}=Ipool;
        
    end
end

hA{1}{1}
Result{1}{1}
pause(5000000)

% menampilkan matrik Ipool
% Result=Ipool;

% disp("Done......!");
