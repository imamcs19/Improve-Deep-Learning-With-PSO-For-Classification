function [y] = Fn_DiffAktivasi(x)
%Hitung Fungsi Aktifasi
%
%  Usage: [y] = Fn_Aktivasi(Xx)
%
%  Parameters: x      - inputs value
%              y      - outputs value
%
%  Author: Imam Cholissodin (imam.cholissodin@gmail.com)

y=Fn_Aktivasi(x).*(1.-Fn_Aktivasi(x));


