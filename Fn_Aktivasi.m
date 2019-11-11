function [y] = Fn_Aktivasi(x)
%Hitung Fungsi Aktifasi
%
%  Usage: [y] = Fn_Aktivasi(Xx)
%
%  Parameters: x      - inputs value
%              y      - outputs value
%
%  Author: Imam Cholissodin (imam.cholissodin@gmail.com)

y=1./(1.+exp(-x));


