clc
clear all
close all
 max = 300;
min = 0;
max2 = 1;
min2 = 0;
a{1} = [
0	0	0	0	0
0	0	131	259	0
0	0	253	95	0
0	0	0	0	0
0	0	0	0	0
];
a{2} = [0	0	0	0	0
0	0	259	95	0
0	0	131	263	0
0	0	0	0	0
0	0	0	0	0
];
a{3} = [0	0	0	0	0
0	0	95	263	0
0	0	259	50	0
0	0	0	0	0
0	0	0	0	0
];
a{4} = [0	0	0	0	0
0	0	263	50	0
0	0	95	154	0
0	0	0	0	0
0	0	0	0	0
];
a{5} = [0	0	0	0	0
0	0	50	154	0
0	0	263	73	0
0	0	0	0	0
0	0	0	0	0
];
a{6} = [0	0	0	0	0
0	0	154	73	0
0	0	50	141	0
0	0	0	0	0
0	0	0	0	0
];
a{7} = [0	0	0	0	0
0	0	73	141	0
0	0	154	146	0
0	0	0	0	0
0	0	0	0	0
];
a{8} = [0	0	0	0	0
0	0	141	146	0
0	0	73	87	0
0	0	0	0	0
0	0	0	0	0
];
 a1 = a{1};
a2 = a{2};
a3 = a{3};
a4 = a{4};
a5 = a{5};
a6 = a{6};
a7 = a{7};
a8 = a{8};
norm{1} = (((a1-min)./(max-min))*(max2-min2))+min2;
norm{2} = (((a2-min)./(max-min))*(max2-min2))+min2;
norm{3} = (((a3-min)./(max-min))*(max2-min2))+min2;
norm{4} = (((a4-min)./(max-min))*(max2-min2))+min2;
norm{5} = (((a5-min)./(max-min))*(max2-min2))+min2;
norm{6} = (((a6-min)./(max-min))*(max2-min2))+min2;
norm{7} = (((a7-min)./(max-min))*(max2-min2))+min2;
norm{8} = (((a8-min)./(max-min))*(max2-min2))+min2;
save('afile.mat','norm')
b1 = norm{1}
b2 = norm{2};
b3 = norm{3};
b4 = norm{4};
b5 = norm{5};
b6 = norm{6};
b7 = norm{7};
b8 = norm{8};
imwrite(b1,'1.jpg')
imwrite(b2,'2.jpg')
imwrite(b3,'3.jpg')
imwrite(b4,'4.jpg')
imwrite(b5,'5.jpg')
imwrite(b6,'6.jpg')
imwrite(b7,'7.jpg')
imwrite(b8,'8.jpg')

% misal ada 3 filter convolution yang digunakan:
% ke-1 (conv11) : average filter
% ke-2 (conv12) : max filter
% ke-3 (conv13) : std filter

% melakukan convolution dengan average filter, max filter,
% std filter
d1_conv11=Function_AvgFilter(b1,3)
d1_conv12=Function_MaxFilter_(b1,3)
d1_conv13=Function_STDFilter(b1,3)

