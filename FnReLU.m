function [Result]=FnReLU(I)

% I=[-0.3049    0.4512    0.4273    0.3788    0.2324
%     0.4573    0.6769    0.6409    0.5682    0.3486
%     0.4573    0.6769    0.6409    0.5682    0.3486
%     0.4573    0.6769    0.6409    0.5682    -0.3486
%     0.3049    0.4512    0.4273    0.3788    0.2324];

% ukuran baris dan kolom dari I
%[mI,nI]=size(I);

% mencari elemen yg nilainya negatif
IdxELneg=find(I<0);

% replace nilai negatif dengan 0
I(IdxELneg)=0;

% menampilkan hasil ReLU
Result=I;


%disp("Done......!");
