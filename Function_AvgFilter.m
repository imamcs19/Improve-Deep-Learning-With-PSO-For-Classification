function [Result]=Function_AvgFilter(I_biner,windows_size)

% windows_size = k 
pad_size=(windows_size-1)/2;

% ukuran baris dan kolom dari I_biner
[mI_biner,nI_biner]=size(I_biner);

% inisialisasi vektor X={x1,x2,.....,xN}
X_=zeros(mI_biner*nI_biner,windows_size^2);

% inisialisasi matrik penampung I_biner
PI_biner=padarray(I_biner,[pad_size pad_size]);

% inisialisasi index matrik PI_biner
Index_PI_biner=1:(mI_biner+(2*pad_size))*(nI_biner+(2*pad_size));

% convert index to xy
[pix,piy] = Index2XY(Index_PI_biner,mI_biner+(2*pad_size));

% mengambil index matrik I_biner yang ada pada matrik PI_biner
xIndex_I_biner_dlm_PI_biner=find(pix>pad_size & pix<(mI_biner+pad_size+1));
yIndex_I_biner_dlm_PI_biner=find(piy>pad_size & piy<(nI_biner+pad_size+1));

% mengambil irisan xIndex_I_biner_dlm_PI_biner dan
% yIndex_I_biner_dlm_PI_biner
xyIndex_I_biner_dlm_PI_biner=intersect(xIndex_I_biner_dlm_PI_biner,yIndex_I_biner_dlm_PI_biner);

ibx=pix(xyIndex_I_biner_dlm_PI_biner);
iby=piy(xyIndex_I_biner_dlm_PI_biner);

% ----------------------------------------------------------------
% membuat penambah_x dan penambah_y otomatis
% ----------------------------------------------------------------
interval_xy=-pad_size:pad_size;

% repeat matrik interval_xy
matrik_repeat_interval_xy=kron(interval_xy,ones(windows_size,1));
penambah_x=sort(matrik_repeat_interval_xy(:));
matrik_repeat_interval_xy=matrik_repeat_interval_xy';
penambah_y=matrik_repeat_interval_xy(:);

for i=1:(windows_size^2)
    % ------------------------------------------------------------------------%
    % untuk X kolom i
        X_i=ibx+penambah_x(i);
        Y_i=iby+penambah_y(i);        
   
    % convert X_i,Y_i menjadi Index
    Index_PI_biner_i= XY2Index(X_i,Y_i,(mI_biner+(2*pad_size)));

    % mengisi X_ kolom i
    X_(:,i)=PI_biner(Index_PI_biner_i);
    % ------------------------------------------------------------------------%
end

%X_'
%sum(X_')
%mean(X_')

Result=reshape(mean(X_')',[mI_biner nI_biner]);

